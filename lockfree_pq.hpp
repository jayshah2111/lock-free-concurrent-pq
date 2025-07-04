```cpp
// lockfree_pq.hpp
#pragma once

#include <atomic>
#include <vector>
#include <mutex>
#include <functional>
#include <random>
#include <limits>
#include <memory>
#include <algorithm>

namespace lf {

// -----------------------------------------------------------------------------
// Simple Hazard Pointer Domain for Memory Reclamation
// -----------------------------------------------------------------------------
class HazardDomain {
    static const size_t MAX_HAZARD_POINTERS = 128;
    std::atomic<void*> hp_[MAX_HAZARD_POINTERS];

    struct Retired {
        void* ptr;
        std::function<void(void*)> deleter;
    };

    std::vector<Retired> retired_;
    std::mutex mtx_;

    HazardDomain() {
        for (size_t i = 0; i < MAX_HAZARD_POINTERS; ++i)
            hp_[i].store(nullptr, std::memory_order_relaxed);
    }

public:
    // Get singleton instance
    static HazardDomain* instance() {
        static HazardDomain inst;
        return &inst;
    }

    // Protect a pointer in a hazard slot (retry until stable)
    template<typename U>
    U* protect(std::atomic<U*>& addr, size_t idx) {
        U* p;
        do {
            p = addr.load(std::memory_order_acquire);
            hp_[idx].store(p, std::memory_order_release);
        } while (p != addr.load(std::memory_order_acquire));
        return p;
    }

    // Retire an object with a custom deleter, recycle when safe
    void retire(void* ptr, std::function<void(void*)> deleter) {
        std::lock_guard<std::mutex> lock(mtx_);
        retired_.push_back({ptr, deleter});
        if (retired_.size() > MAX_HAZARD_POINTERS) scan();
    }

    // Scan hazard pointers and reclaim safe-to-delete nodes
    void scan() {
        std::vector<void*> hazards;
        hazards.reserve(MAX_HAZARD_POINTERS);
        for (size_t i = 0; i < MAX_HAZARD_POINTERS; ++i) {
            void* p = hp_[i].load(std::memory_order_acquire);
            if (p) hazards.push_back(p);
        }
        auto it = retired_.begin();
        while (it != retired_.end()) {
            if (std::find(hazards.begin(), hazards.end(), it->ptr) == hazards.end()) {
                it->deleter(it->ptr);
                it = retired_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Lock-Free Concurrent Min-Priority Queue (Skiplist-based)
// -----------------------------------------------------------------------------
template<typename T>
class LockFreePQ {
private:
    // Maximum levels for skiplist
    static const int MaxLevel = 16;
    static constexpr double Probability = 0.5;

    struct Node {
        T value;
        int topLevel;
        std::atomic<Node*> next[MaxLevel + 1];
        std::atomic<bool> marked;
        std::atomic<bool> fullyLinked;

        // Sentinel constructor
        Node(int level)
            : value(), topLevel(level), marked(false), fullyLinked(false)
        {
            for (int i = 0; i <= level; ++i)
                next[i].store(nullptr, std::memory_order_relaxed);
        }

        // Value node constructor
        Node(const T& val, int level)
            : value(val), topLevel(level), marked(false), fullyLinked(false)
        {
            for (int i = 0; i <= level; ++i)
                next[i].store(nullptr, std::memory_order_relaxed);
        }
    };

    // Head and tail sentinels
    Node* head_;
    Node* tail_;
    std::atomic<size_t> count_;
    HazardDomain* domain_;

    // Random number generator for levels
    static thread_local std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};

    // Generate random level
    int randomLevel() {
        int lvl = 0;
        while (dist_(rng_) < Probability && lvl < MaxLevel)
            ++lvl;
        return lvl;
    }

    // Find preds and succs for a given key
    bool findNode(const T& key, Node* preds[], Node* succs[]) {
        bool found = false;
        Node* pred = head_;
        for (int level = MaxLevel; level >= 0; --level) {
            Node* curr = pred->next[level].load(std::memory_order_acquire);
            while (true) {
                Node* succ = curr->next[level].load(std::memory_order_acquire);
                // Skip over marked nodes
                while (curr->marked.load(std::memory_order_acquire)) {
                    if (!pred->next[level].compare_exchange_strong(
                            curr, succ,
                            std::memory_order_acq_rel)) {
                        curr = pred->next[level].load(std::memory_order_acquire);
                        continue;
                    }
                    curr = succ;
                    succ = curr->next[level].load(std::memory_order_acquire);
                }
                if (curr == tail_ || curr->value < key) {
                    pred = curr;
                    curr = succ;
                } else {
                    break;
                }
            }
            preds[level] = pred;
            succs[level] = curr;
        }
        if (succs[0] != tail_ && succs[0]->value == key)
            found = true;
        return found;
    }

public:
    // Construct priority queue
    LockFreePQ(HazardDomain* domain = nullptr)
        : count_(0)
    {
        domain_ = domain ? domain : HazardDomain::instance();
        head_ = new Node(MaxLevel);
        tail_ = new Node(MaxLevel);
        for (int i = 0; i <= MaxLevel; ++i)
            head_->next[i].store(tail_, std::memory_order_relaxed);
        // Seed RNG per thread
        std::random_device rd;
        rng_.seed(rd());
    }

    ~LockFreePQ() {
        // Delete all nodes
        Node* node = head_;
        while (node) {
            Node* next = node->next[0].load(std::memory_order_relaxed);
            delete node;
            node = (next == tail_) ? nullptr : next;
        }
        delete tail_;
    }

    // Disable copy
    LockFreePQ(const LockFreePQ&) = delete;
    LockFreePQ& operator=(const LockFreePQ&) = delete;

    // Push an item (multiple producers)
    void push(const T& item) noexcept {
        Node* preds[MaxLevel + 1];
        Node* succs[MaxLevel + 1];
        int topLevel = randomLevel();
        while (true) {
            findNode(item, preds, succs);
            Node* newNode = new Node(item, topLevel);
            for (int lvl = 0; lvl <= topLevel; ++lvl)
                newNode->next[lvl].store(succs[lvl], std::memory_order_relaxed);
            Node* pred = preds[0];
            Node* succ = succs[0];
            if (!pred->next[0].compare_exchange_strong(
                    succ, newNode,
                    std::memory_order_acq_rel)) {
                delete newNode;
                continue;
            }
            for (int lvl = 1; lvl <= topLevel; ++lvl) {
                while (!preds[lvl]->next[lvl].compare_exchange_strong(
                        succs[lvl], newNode,
                        std::memory_order_acq_rel)) {
                    findNode(item, preds, succs);
                }
            }
            newNode->fullyLinked.store(true, std::memory_order_release);
            count_.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }

    void push(T&& item) noexcept {
        push(item);
    }

    // Pop minimum item (multiple consumers)
    bool pop(T& out) noexcept {
        Node* node = nullptr;
        while (true) {
            node = head_->next[0].load(std::memory_order_acquire);
            if (node == tail_)
                return false;
            if (!node->fullyLinked.load(std::memory_order_acquire))
                continue;
            bool expected = false;
            if (node->marked.compare_exchange_strong(
                    expected, true,
                    std::memory_order_acq_rel)) {
                out = node->value;
                Node* preds[MaxLevel + 1];
                Node* succs[MaxLevel + 1];
                findNode(node->value, preds, succs);
                for (int lvl = node->topLevel; lvl >= 0; --lvl) {
                    preds[lvl]->next[lvl].compare_exchange_strong(
                        node, node->next[lvl].load(std::memory_order_acquire),
                        std::memory_order_acq_rel);
                }
                domain_->retire(node, [](void* p) {
                    delete static_cast<Node*>(p);
                });
                count_.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
        }
    }

    // Check if empty (approximate under concurrency)
    bool empty() const noexcept {
        return count_.load(std::memory_order_relaxed) == 0;
    }

    // Approximate size
    size_t size() const noexcept {
        return count_.load(std::memory_order_relaxed);
    }
};

// Thread-local RNG initialization
template<typename T>
thread_local std::mt19937_64 LockFreePQ<T>::rng_;

} // namespace lf
```
