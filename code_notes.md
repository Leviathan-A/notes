# 1.1斜着遍历

```cpp
for(int l=1;l<=n;l++)
{
    for(int i=0;i<=n-l;i++)
    {
        int j=l+i-1;
        dp[i][j];
	}
}

int i,j;
      for(int num = 0;num<m+n-1;num++)
       {
            for(i=0;i<m;i++)
        	{
             j=num-i;
             if(j>=0 && j<n)
               cout<<vec[i][j]<<" ";
        	}
        	cout<<endl;
       }

```

```cpp
int[][] dp = new int[m][n];
for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
        // 计算 dp[i][j]
```

```cpp
for (int i = m - 1; i >= 0; i--)
    for (int j = n - 1; j >= 0; j--)
        // 计算 dp[i][j]
```

```cpp
// 斜着遍历数组
for (int l = 2; l <= n; l++) {
    for (int i = 0; i <= n - l; i++) {
        int j = l + i - 1;
        // 计算 dp[i][j]
    }
}
```

# 1.2dfs和bfs

### 最短用bfs，求方案dfs

### dfs遍历框架(回溯)

```java
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

示范：全排列

```java
List<List<Integer>> res = new LinkedList<>();

/* 主函数，输入一组不重复的数字，返回它们的全排列 */
List<List<Integer>> permute(int[] nums) {
    // 记录「路径」
    LinkedList<Integer> track = new LinkedList<>();
    backtrack(nums, track);
    return res;
}

// 路径：记录在 track 中
// 选择列表：nums 中不存在于 track 的那些元素
// 结束条件：nums 中的元素全都在 track 中出现
void backtrack(int[] nums, LinkedList<Integer> track) {
    // 触发结束条件
    if (track.size() == nums.length) {
        res.add(new LinkedList(track));
        return;
    }

    for (int i = 0; i < nums.length; i++) {
        // 排除不合法的选择
        if (track.contains(nums[i]))
            continue;
        // 做选择
        track.add(nums[i]);
        // 进入下一层决策树
        backtrack(nums, track);
        // 取消选择
        track.removeLast();
    }
}
```

### bfs框架

```java
// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
    Queue<Node> q; // 核心数据结构
    Set<Node> visited; // 避免走回头路

    q.offer(start); // 将起点加入队列
    visited.add(start);
    int step = 0; // 记录扩散的步数

    while (q not empty) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            /* 划重点：这里判断是否到达终点 */
            if (cur is target)
                return step;
            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj())
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
        }
        /* 划重点：更新步数在这里 */
        step++;
    }
}
```

示范：树的深度

```java
int minDepth(TreeNode root) {
    if (root == null) return 0;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    // root 本身就是一层，depth 初始化为 1
    int depth = 1;

    while (!q.isEmpty()) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            TreeNode cur = q.poll();
            /* 判断是否到达终点 */
            if (cur.left == null && cur.right == null) 
                return depth;
            /* 将 cur 的相邻节点加入队列 */
            if (cur.left != null)
                q.offer(cur.left);
            if (cur.right != null) 
                q.offer(cur.right);
        }
        /* 这里增加步数 */
        depth++;
    }
    return depth;
}
```

# 1.3并查集（真假判断，有逻辑传递性）

```c++
class UnionFind {
private:
    vector<int> parent;

public:
    UnionFind() {
        parent.resize(26);
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int index) {
        if (index == parent[index]) {
            return index;
        }
        parent[index] = find(parent[index]);
        return parent[index];
    }

    void unite(int index1, int index2) {
        parent[find(index1)] = find(index2);
    }
};
```

# 1.4LRU和LFU

### LRU

```c++
struct DLinkedNode {
    int key, value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(int _key, int _value): key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

public:
    LRUCache(int _capacity): capacity(_capacity), size(0) {
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if (!cache.count(key)) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (!cache.count(key)) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = removeTail();
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                // 防止内存泄漏
                delete removed;
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }
    
    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};

```

### LFU

```c++
// 缓存的节点信息
struct Node {
    int key, val, freq;
    Node(int _key,int _val,int _freq): key(_key), val(_val), freq(_freq){}
};
class LFUCache {
    int minfreq, capacity;
    unordered_map<int, list<Node>::iterator> key_table;
    unordered_map<int, list<Node>> freq_table;
public:
    LFUCache(int _capacity) {
        minfreq = 0;
        capacity = _capacity;
        key_table.clear();
        freq_table.clear();
    }
    
    int get(int key) {
        if (capacity == 0) return -1;
        auto it = key_table.find(key);
        if (it == key_table.end()) return -1;
        list<Node>::iterator node = it -> second;
        int val = node -> val, freq = node -> freq;
        freq_table[freq].erase(node);
        // 如果当前链表为空，我们需要在哈希表中删除，且更新minFreq
        if (freq_table[freq].size() == 0) {
            freq_table.erase(freq);
            if (minfreq == freq) minfreq += 1;
        }
        // 插入到 freq + 1 中
        freq_table[freq + 1].push_front(Node(key, val, freq + 1));
        key_table[key] = freq_table[freq + 1].begin();
        return val;
    }
    
    void put(int key, int value) {
        if (capacity == 0) return;
        auto it = key_table.find(key);
        if (it == key_table.end()) {
            // 缓存已满，需要进行删除操作
            if (key_table.size() == capacity) {
                // 通过 minFreq 拿到 freq_table[minFreq] 链表的末尾节点
                auto it2 = freq_table[minfreq].back();
                key_table.erase(it2.key);
                freq_table[minfreq].pop_back();
                if (freq_table[minfreq].size() == 0) {
                    freq_table.erase(minfreq);
                }
            } 
            freq_table[1].push_front(Node(key, value, 1));
            key_table[key] = freq_table[1].begin();
            minfreq = 1;
        } else {
            // 与 get 操作基本一致，除了需要更新缓存的值
            list<Node>::iterator node = it -> second;
            int freq = node -> freq;
            freq_table[freq].erase(node);
            if (freq_table[freq].size() == 0) {
                freq_table.erase(freq);
                if (minfreq == freq) minfreq += 1;
            }
            freq_table[freq + 1].push_front(Node(key, value, freq + 1));
            key_table[key] = freq_table[freq + 1].begin();
        }
    }
};
```

手工实现list版

```c++
class LFUCache {
private:
    struct DLNode {
        int key;
        int value;
        int freq;
        DLNode* prev;
        DLNode* next;
        DLNode() : key(-1), value(-1), freq(0), prev(NULL), next(NULL) {};
        DLNode(int k, int v, int f) : key(k), value(v), freq(f), prev(NULL), next(NULL) {};
    };
    struct DLList {
        DLNode* head;
        DLNode* tail;
        int size;
        DLList() {
            head = new DLNode;
            tail = new DLNode;
            head->next = tail;
            tail->prev = head;
            size = 0;
        }
        void remove(DLNode* node) {
            node->next->prev = node->prev;
            node->prev->next = node->next;
            --size;
        }
        DLNode* pop_tail() {
            auto node = tail->prev;
            remove(node);
            return node;
        }
        void add_front(DLNode* node) {
            node->prev = head;
            node->next = head->next;
            node->prev->next = node;
            node->next->prev = node;
            ++size;
        }
        ~DLList() {
            size = 0;
            delete head;
            delete tail;
        }
    };
    int capacity;
    int size;
    map<int, DLList> freqDict;
    map<int, DLNode*> cache;
    
    void incr(DLNode* node) {
        freqDict[node->freq].remove(node);
        if ((freqDict[node->freq].size) == 0) {
            freqDict.erase(node->freq);
        }
        ++node->freq;
        freqDict[node->freq].add_front(node);
    }
    
    void add(DLNode* node) {
        cache[node->key] = node;
        freqDict[node->freq].add_front(node);
        ++size;
    }
    
    void pop() {
        if (freqDict.empty()) return;
        auto it = freqDict.begin();
        auto node = it->second.pop_tail();
        if (it->second.size == 0) {
            freqDict.erase(it);
        }
        cache.erase(node->key);
        delete node;
        --size;
    }

public:    
    LFUCache(int capacity) {
        this->capacity = capacity;
        this->size = 0;
    }
    
    int get(int key) {
        if (cache.count(key) == 0) return -1;
        auto node = cache[key];
        incr(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (capacity == 0) return;
        if (cache.count(key) == 0) {
            if (size == capacity) pop();
            auto node = new DLNode(key, value, 1);
            add(node);
        } else {
            auto node = cache[key];
            node->value = value;
            incr(node);
        }
    }
};
```

# 1.5优先队列使用

寻找中位数

```c++
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int> hi;
    priority_queue<int, vector<int>, greater<int> > lo;
    MedianFinder() {

    }
    
    void addNum(int num) {
        hi.push(num);
        lo.push(hi.top());
        hi.pop();
        if(hi.size()<lo.size())
        {
            hi.push(lo.top());
            lo.pop();
        }
    }
    
    double findMedian() {
        if(hi.size()==lo.size())return (hi.top()+lo.top())*0.5;
        else return hi.top();
    }
};

```

# 1.6单调栈（Next Great Number）（尽量放id）

模板

```c++
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        vector<int> ans;
        unordered_map<int, int> num_table;
        stack<int> s;
        for(int i=nums2.size()-1;i>=0; --i)
        {
            while(!s.empty() && nums2[i]>s.top())
            {
                s.pop();
            }
            num_table[nums2[i]] = s.empty()?-1:s.top();
            s.push(nums2[i]);
        }
        for(int i:nums1)ans.push_back(num_table[i]);
        return ans;
    }
};
```

# 1.7滑动窗口

## 1.7.1处理数字

### 优先队列（对带id的pair用优先队列进行排序）

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        priority_queue<pair<int, int>> q;
        for(int i=0; i<k; ++i)
        {
            q.emplace(nums[i],i);
        }

        vector<int> ans = {q.top().first};
        for(int i=k; i<nums.size(); ++i)
        {
            q.emplace(nums[i],i);
            while(q.top().second<=i-k)q.pop();
            ans.push_back(q.top().first);
        }
        return ans;
    }
};
```

### 单调队列（使用队列两头拍扁）

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        deque<int> q;
        for (int i = 0; i < k; ++i) {
            while (!q.empty() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
        }

        vector<int> ans = {nums[q.front()]};
        for (int i = k; i < n; ++i) {
            while (!q.empty() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
            while (q.front() <= i - k) {
                q.pop_front();
            }
            ans.push_back(nums[q.front()]);
        }
        return ans;
    }
};
```

## 1.7.2双指针

```c++
/* 滑动窗口算法框架 */
void slidingWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0; 
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 右移窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...

        /*** debug 输出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/

        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            // 左移窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
```

1、当移动 `right` 扩大窗口，即加入字符时，应该更新哪些数据？

2、什么条件下，窗口应该暂停扩大，开始移动 `left` 缩小窗口？

3、当移动 `left` 缩小窗口，即移出字符时，应该更新哪些数据？

4、我们要的结果应该在扩大窗口时还是缩小窗口时进行更新？

# 1.8二分查找

### 找一个数（长度-1；小于等于；返回目标）

```java
int binary_search(int[] nums, int target) {
    int left = 0, right = nums.length - 1; 
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1; 
        } else if(nums[mid] == target) {
            // 直接返回
            return mid;
        }
    }
    // 直接返回
    return -1;
}
```

### 找左侧边界（长度-1；小于等于；返回左侧）

```java
int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    // 搜索区间为 [left, right]
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            // 搜索区间变为 [mid+1, right]
            left = mid + 1;
        } else if (nums[mid] > target) {
            // 搜索区间变为 [left, mid-1]
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 收缩右侧边界
            right = mid - 1;
        }
    }
    // 检查出界情况
    if (left >= nums.length || nums[left] != target)
        return -1;
    return left;
}
```

### 找右侧边界（长度-1；小于等于；返回右侧）

```java
int right_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，锁定右侧边界
            left = mid + 1;
        }
    }
    // 最后要检查 right 越界的情况
    if (right < 0 || nums[right] != target)
        return -1;
    return right;
}
```

这些问题都如出一辙，请大家特别留意题目中出现的关键字**「非负整数」、分割「连续」**，思考清楚设计算法的关键步骤和原因，相信以后遇到类似的问题就能轻松应对。

「力扣」第 875 题：爱吃香蕉的珂珂（中等）
LCP 12. 小张刷题计划 （中等）
「力扣」第 1482 题：制作 m 束花所需的最少天数（中等）
「力扣」第 1011 题：在 D 天内送达包裹的能力（中等）
「力扣」第 1552 题：两球之间的磁力（中等）
总结
这道题让我们**「查找一个有范围的整数」**，以后遇到类似问题，要想到可以尝试使用「二分」；
遇到类似使**「最大值」最小化**，这样的题目描述，可以好好跟自己做过的这些问题进行比较，看看能不能找到关联；
在代码层面上，这些问题的特点都是：**在二分查找的判别函数里，需要遍历数组一次。**

# 1.9双指针

我把双指针技巧再分为两类，**一类是「快慢指针」，一类是「左右指针」**。前者解决主要解决**链表**中的问题，比如典型的判定链表中是否包含环；后者主要解决**数组（或者字符串）**中的问题，比如二分查找。

[141.环形链表（简单）](https://leetcode-cn.com/problems/linked-list-cycle)

[142.环形链表II（简单）](https://leetcode-cn.com/problems/linked-list-cycle-ii)

[167.两数之和 II - 输入有序数组（中等）](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted)

[344.反转字符串（简单）](https://leetcode-cn.com/problems/reverse-string/)

[19.删除链表倒数第 N 个元素（中等）](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list)



# 2.1小技巧

### stl找最大max——element

### stl求和accumulate(vec.begin() , vec.end() , 42);

### 整数个数time += (p - 1) / K + 1;

### 
