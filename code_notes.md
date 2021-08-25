# 1.数据结构

## 1.1斜着遍历

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

1、遍历的过程中，所需的状态必须是已经计算出来的**。

**2、遍历的终点必须是存储结果的那个位置**。

主要就是看 base case 和最终结果的存储位置，保证遍历过程中使用的数据都是计算完毕的就行



**递归遍历数组**

[698.划分为k个相等的子集（中等）](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/)

```c++
for (int i = start; i < nums.length; i++) {
        // 剪枝
        if (used[i]) {
            // nums[i] 已经被装入别的桶中
            continue;
        }
        if (nums[i] + bucket > target) {
            // 当前桶装不下 nums[i]
            continue;
        }
        // 做选择，将 nums[i] 装入当前桶中
        used[i] = true;
        bucket += nums[i];
        // 递归穷举下一个数字是否装入当前桶
        if (backtrack(k, bucket, nums, i + 1, used, target)) {
            return true;
        }
        // 撤销选择
        used[i] = false;
        bucket -= nums[i];
    }
```

**递归遍历矩阵的某一个区域**

[37.解数独（困难）](https://leetcode-cn.com/problems/sudoku-solver)

```c++
    bool valid(vector<vector<char>>& board,int row,int col,char ch)
    {
        for (int i = 0; i < 9; i++) {
        // 判断行是否存在重复
        if (board[row][i] == ch) return false;
        // 判断列是否存在重复
        if (board[i][col] == ch) return false;
        // 判断 3 x 3 方框是否存在重复
        if (board[(row/3)*3 + i/3][(col/3)*3 + i%3] == ch)
            return false;
    }
```

**单次斜着遍历**

```c++
for (int i = row - 1, j = col - 1;
            i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q')
            return false;
```



## 1.2dfs和bfs

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

[698.划分为k个相等的子集（中等）](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/)

### 回溯视角问题

先说第一个解法，也就是从数字的角度进行穷举，`n` 个数字，每个数字有 `k` 个桶可供选择，所以组合出的结果个数为 `k^n`，时间复杂度也就是 `O(k^n)`。

第二个解法，每个桶要遍历 `n` 个数字，选择「装入」或「不装入」，组合的结果有 `2^n` 种；而我们有 `k` 个桶，所以总的时间复杂度为 `O(k*2^n)`。

**当然，这是理论上的最坏复杂度，实际的复杂度肯定要好一些，毕竟我们添加了这么多剪枝逻辑**。不过，从复杂度的上界已经可以看出第一种思路要慢很多了。

所以，谁说回溯算法没有技巧性的？虽然回溯算法就是暴力穷举，但穷举也分聪明的穷举方式和低效的穷举方式，关键看你以谁的「视角」进行穷举。

通俗来说，我们应该尽量「少量多次」，就是说宁可多做几次选择，也不要给太大的选择空间；宁可「二选一」选 `k` 次，也不要 「`k` 选一」选一次。

### 走格子示范（小红书笔试）

```c++
#include<iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include<string>
#include<numeric>
#include <sstream>
#include <iomanip>
#include<unordered_map>
#include<set>

using namespace std;
vector<vector<int>>dir{ {0,1},{0,-1},{1,0},{-1,0} };
int res = 0;
void dfs(vector<string>& chess, int size, int i, int j) {
	if (size - 1 == 0) {
		if (i == chess.size() - 1) {
			res++;
		}
		return;
	}
	for (int k = 0; k < 4; ++k) {
		int ni = i + dir[k][0];
		int nj = j + dir[k][1];
		if (ni < 0 || ni >= chess.size() || nj < 0 || nj >= chess[0].size() || chess[ni][nj] != '.')continue;
		chess[i][j] = '#';
		size--;
		dfs(chess, size, ni, nj);
		size++;
		chess[i][j] = '.';
	}
}

/*
3
#.#
..#
...

*/
int main() {
	int N;
	cin >> N;
	string str;
	vector<string>vec;
	for (int i = 0; i < N; ++i) {
		cin >> str;
		vec.push_back(str);
	}
	int size = 0;
	for (int i = 0; i < vec.size(); ++i) {
		for (int j = 0; j < vec[i].size(); ++j) {
			if (vec[i][j] == '.')size++;
		}
	}
	if (vec.size() == 0 || vec[0].size() == 0)return 0;
	//if (vec[0][0] == '#')size;

	dfs(vec, size, 0, 0);
	cout << res << endl;
	return 0;
}

```



## 1.2.5bfs框架

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

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int minDepth(TreeNode* root) {
        if(root==NULL)return 0;
        deque<TreeNode*> q;
        q.push_front(root);
        int depth=1;
        while(!q.empty())
        {
            int n=q.size();
            for(int i=0;i<n;i++)
            {
                TreeNode* tmp=q.front();
                q.pop_front();
                if(tmp->left==NULL && tmp->right==NULL)
                return depth;
                if(tmp->left)q.push_back(tmp->left);
                if(tmp->right)q.push_back(tmp->right);
            }
            depth++;
        }
        return depth;

    }
};
```

**双向bfs**

[752.打开转盘锁（中等）](https://leetcode-cn.com/problems/open-the-lock)

```c++
class Solution {
public:
    string UP(string tmp,int j)
    {
        if(tmp[j]=='9')tmp[j]='0';
        else tmp[j]+=1;
        return tmp;
    }
    string DOWN(string tmp,int j)
    {
        if(tmp[j]=='0')tmp[j]='9';
        else tmp[j]-=1;
        return tmp;
    }
    int openLock(vector<string>& deadends, string target) {
        set<string> q1;
        set<string> q2;
        set<string> d;
        set<string> used;
        for(auto s:deadends)d.insert(s);
        q1.insert("0000");
        q2.insert(target);
        //used.insert("0000");
        int step=0;
        while(!q1.empty()&&!q2.empty())
        {
            set<string> temp;
            for(auto cur:q1)
            {
                if(d.count(cur))continue;
                if(q2.count(cur))return step;
                used.insert(cur);
                for(int i=0;i<4;i++)
                {
                    string tmp=UP(cur,i);
                    if(!used.count(tmp))
                    {
                        temp.insert(tmp);
                    }
                    string tmp1 = DOWN(cur,i);
                    if(!used.count(tmp1))
                    temp.insert(tmp1);
                }
            }
            step++;
            q1=q2;
            q2=temp;
        }
        return -1;
    }
};
```

如何转化bfs，以及矩阵换位的处理，如何转字符串处理；

[773.滑动谜题（困难）](https://leetcode-cn.com/problems/sliding-puzzle)

```c++
class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        int m=2,n=3;
        string start="";
        string end="123450";
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                start.push_back(board[i][j]+'0');
            }
        }
        vector<vector<int>> neighbor = {
        { 1, 3 },
        { 0, 4, 2 },
        { 1, 5 },
        { 0, 4 },
        { 3, 1, 5 },
        { 4, 2 }
        };

        deque<string> q;
        set<string> used;

        q.push_back(start);
        used.insert(start);
        int step=0;
        while(!q.empty())
        {
            int n=q.size();
            for(int i=0;i<n;i++)
            {
                string tmp=q.front();
                q.pop_front();
                if(end==tmp)return step;
                // used.insert(tmp);
                int id=0;
                while(tmp[id]!='0')id++;
                for(int adj:neighbor[id])
                {
                    string newboard=tmp;
                    swap(newboard[adj],newboard[id]);
                    if(!used.count(newboard))
                    {
                        used.insert(newboard);
                        q.push_back(newboard);
                    }

                }

            }
            step++;
        }
        return -1;
    }
};
```



## 1.3并查集（真假判断，有逻辑传递性）

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

## 1.4LRU和LFU

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

## 1.5优先队列使用

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

## 1.6单调栈（Next Great Number）（尽量放id）

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

字符串去重的变体，此处大小为字符串的字典顺序

```c++
class Solution {
public:
 string removeDuplicateLetters(string s) {
	int dict[26] = { 0 };
	int in[26] = { 0 };
	for (char c : s)
		dict[c - 'a']++;
	stack<char> st;
	// st.push('a');
	string ans;
	for(char c:s)
    {
        dict[c-'a']--;
        if(in[c-'a'])continue;
        while(!st.empty() && st.top() > c)
        {
            if(dict[st.top()-'a']==0)break;
            in[st.top()-'a']--;
            st.pop();
        }
        st.push(c);
        in[c-'a']=1;

    }
	while (!st.empty())
	{
		ans.push_back(st.top());
		st.pop();
	}
    reverse(ans.begin(),ans.end());
	return ans;
    }
};
```



## 1.7滑动窗口

### 1.7.1处理数字

#### 优先队列（对带id的pair用优先队列进行排序）

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

#### 单调队列（使用队列两头拍扁）

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

### 1.7.2双指针

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

## 1.8二分查找

#### 找一个数（长度-1；小于等于；返回目标）

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

#### 找左侧边界（长度-1；小于等于；返回左侧）

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

#### 找右侧边界（长度-1；小于等于；返回右侧）

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

## 1.9双指针（非滑动窗口）

我把双指针技巧再分为两类，**一类是「快慢指针」，一类是「左右指针」**。前者解决主要解决**链表**中的问题，比如典型的判定链表中是否包含环；后者主要解决**数组（或者字符串）**中的问题，比如二分查找。

[141.环形链表（简单）](https://leetcode-cn.com/problems/linked-list-cycle)

[142.环形链表II（简单）](https://leetcode-cn.com/problems/linked-list-cycle-ii)

[167.两数之和 II - 输入有序数组（中等）](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted)

[344.反转字符串（简单）](https://leetcode-cn.com/problems/reverse-string/)

[19.删除链表倒数第 N 个元素（中等）](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list)

#### 快慢指针处理重复元素

```java
int removeDuplicates(int[] nums) {
    if (nums.length == 0) {
        return 0;
    }
    int slow = 0, fast = 0;
    while (fast < nums.length) {
        if (nums[fast] != nums[slow]) {
            slow++;
            // 维护 nums[0..slow] 无重复
            nums[slow] = nums[fast];
        }
        fast++;
    }
    // 数组长度为索引 + 1
    return slow + 1;
}
```



# 2.动态规划

动态规划问题的一般形式就是**求最值**，求**最长**递增子序列呀，**最小**编辑距离呀等等。

求解动态规划的核心问题是**穷举**

由dp[i-1]无法推断出dp[i]时，考虑以dp[i]为结尾，再搜索dp数组来解题。

**存在「重叠子问题」**，如果暴力穷举的话效率会极其低下，所以需要**「备忘录」或者「DP table」**来优化穷举过程，避免不必要的计算。

而且，动态规划问题一定会**具备「最优子结构」**，才能通过子问题的最值得到原问题的最值

**明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义**

```java
# 初始化 base case
dp[0][0][...] = base
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
```

## 2.1动态规划与回溯对比

回溯框架（可见1.2）

动态规划问题的一般形式就是**求最值**，求**最长**递增子序列呀，**最小**编辑距离呀等等。回溯是求**所有方案**

```python
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
        
def backtrack(nums, i):
    if i == len(nums):
        if 达到 target:
            result += 1
        return

    for op in { +1, -1 }:
        选择 op * nums[i]
        # 穷举 nums[i + 1] 的选择
        backtrack(nums, i + 1)
        撤销选择
```

# 2.2自顶向下dp和自底向上dp

**DP table 是自底向上求解，递归解法是自顶向下求解**：

编辑距离用递归**自顶向下+备忘录**

```c++
class Solution {
public:
    
    int dp(string& word1, string& word2, int i, int j,vector<vector<int>>& memo)
    {
        if(i==-1)return j+1;
        if(j==-1)return i+1;
        if(memo[i][j]!=-1)return memo[i][j];
        if(word1[i]==word2[j])
        {
            memo[i][j]=dp(word1,word2,i-1,j-1,memo);
            return memo[i][j];
        }
        else
        {
        memo[i][j]=min(dp(word1,word2,i-1,j,memo)+1,min(dp(word1,word2,i-1,j-1,memo)+1,dp(word1,word2,i,j-1,memo)+1));
        return memo[i][j];

        }
    }
    int minDistance(string word1, string word2) {
        vector<vector<int>> memo(word1.size(),vector<int>(word2.size(),-1));
        return dp(word1,word2,word1.size()-1,word2.size()-1,memo);
    }
};
```

编辑距离用递归DP_TABLE自底向上

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        vector<vector<int>> dp(word1.size()+1,vector<int>(word2.size()+1,0));
        for(int i=0;i<=word1.size();i++)
        {
            dp[i][0]=i;
        }
        for(int j=0;j<=word2.size();j++)
        {
            dp[0][j]=j;
        }
        for(int i=1;i<=word1.size();i++)
        {
            for(int j=1;j<=word2.size();j++)
            {
                if(word1[i-1]==word2[j-1])
                dp[i][j]=dp[i-1][j-1];
                else
                dp[i][j]=min(dp[i-1][j]+1,min(dp[i-1][j-1]+1,dp[i][j-1]+1));
            }
        }
        return dp[word1.size()][word2.size()];
    }
};
```

**搞不清楚dp_table或者dp_table不好递推的时候用递归+备忘录。**

**dp_table定义容易且递推容易时用数组遍历**。

## 2.3最大上升序列

套娃信封

```c++
class Solution {
public:
    static bool cmp(vector<int>v1,vector<int>v2)
    {
        if(v1[0]<v2[0])return true;
        else if(v1[0]==v2[0])
        return v1[1]>v2[1]?true:false;
        else return false;
    }
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(),envelopes.end(),cmp);
        vector<int> dp(envelopes.size(),1);
        for(int i=1;i<envelopes.size();i++)
        {
            for(int j=0;j<i;j++)
            {
                if(envelopes[i][1]>envelopes[j][1])
                dp[i]=max(dp[j]+1,dp[i]);
            }
        }
        return *max_element(dp.begin(),dp.end());
    }
};
```

## 2.4子序列问题

**1、第一种思路模板是一个一维的 dp 数组：**

```java
int n = array.length;
int[] dp = new int[n];

for (int i = 1; i < n; i++) {
    for (int j = 0; j < i; j++) {
        dp[i] = 最值(dp[i], dp[j] + ...)
    }
}
```

举个我们写过的例子 [最长递增子序列](http://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484498&idx=1&sn=df58ef249c457dd50ea632f7c2e6e761&chksm=9bd7fa5aaca0734c29bcf7979146359f63f521e3060c2acbf57a4992c887aeebe2a9e4bd8a89&scene=21#wechat_redirect)，在这个思路中 dp 数组的定义是：

**在子数组`array[0..i]`中，以\**`array[i]`\**结尾的目标子序列（最长递增子序列）的长度是`dp[i]`**。

为啥最长递增子序列需要这种思路呢？前文说得很清楚了，因为这样符合归纳法，可以找到状态转移的关系，这里就不具体展开了。

**2、第二种思路模板是一个二维的 dp 数组：**

```java
int n = arr.length;
int[][] dp = new dp[n][n];

for (int i = 0; i < n; i++) {
    for (int j = 1; j < n; j++) {
        if (arr[i] == arr[j]) 
            dp[i][j] = dp[i][j] + ...
        else
            dp[i][j] = 最值(...)
    }
}
```

这种思路运用相对更多一些，尤其是涉及两个字符串/数组的子序列。本思路中 dp 数组含义又分为「只涉及一个字符串」和「涉及两个字符串」两种情况。

**2.1** **涉及两个字符串/数组时**（比如最长公共子序列），dp 数组的含义如下：

**在子数组`arr1[0..i]`和子数组`arr2[0..j]`中，我们要求的子序列（最长公共子序列）长度为`dp[i][j]`**。

**2.2** **只涉及一个字符串/数组时**（比如本文要讲的最长回文子序列），dp 数组的含义如下：

**在子数组`array[i..j]`中，我们要求的子序列（最长回文子序列）的长度为`dp[i][j]`**。

另外，找到状态转移和 base case 之后，**一定要观察 DP table**，看看怎么遍历才能保证通过已计算出来的结果解决新的问题

## 2.5背包问题

[背包九讲]: https://www.bilibili.com/video/av33930433

### 1.01背包

[01背包]: https://www.acwing.com/problem/content/2/

二维dp表

| 0    | 0    | 0    | 0    | 0    |
| ---- | ---- | ---- | ---- | ---- |
| 0    | 2    | 2    | 2    | 2    |
| 0    | 2    | 4    | 4    | 4    |
| 0    | 2    | 6    | 6    | 6    |
| 0    | 2    | 6    | 6    | 6    |
| 0    | 2    | 6    | 8    | 8    |



```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    vector<vector<int>> data(N,vector<int>(2));
    for(int i=0;i<N;i++)
    cin>>data[i][0]>>data[i][1];
    
    vector<vector<int>> dp(N+1,vector<int>(V+1,0));
    
    for(int i=1;i<=N;i++)
    {
        for(int j=1;j<=V;j++)
        {
            if(j-data[i-1][0]<0)
            dp[i][j]=dp[i-1][j];
            else
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-data[i-1][0]]+data[i-1][1]);
        }
    }
    cout<<dp[N][V];
    return 0;
}
```

一维dp表

| 0    | 2    | 2    | 2    | 2    |
| ---- | ---- | ---- | ---- | ---- |
| 0    |      | 4    | 6    | 6    |
| 0    |      |      | 6    | 8    |
| 0    |      |      |      | 8    |

```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    vector<vector<int>> data(N,vector<int>(2));
    for(int i=0;i<N;i++)
    cin>>data[i][0]>>data[i][1];
    
    vector<int> dp(V+1,0);
    
    for(int i=1;i<=N;i++)
    {
        for(int j=V;j>=data[i-1][0];j--)
            dp[j]=max(dp[j],dp[j-data[i-1][0]]+data[i-1][1]);
    }
    cout<<dp[V];
    return 0;
}
```



### 2.完全背包问题

[完全背包]: https://www.acwing.com/problem/content/3/



```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    vector<vector<int>> data(N,vector<int>(2));
    for(int i=0;i<N;i++)
    cin>>data[i][0]>>data[i][1];
    
    vector<int> dp(V+1,0);
    
    for(int i=1;i<=N;i++)
    {
        for(int j=data[i-1][0];j<=V;j++)
            dp[j]=max(dp[j],dp[j-data[i-1][0]]+data[i-1][1]);
    }
    cout<<dp[V];
    return 0;
}
```



输入优化

```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    int v,w;
    vector<int> dp(V+1,0);
    for(int i=0;i<N;i++)
    {
        cin>>v>>w;
        for(int j=v;j<=V;j++)
        dp[j]=max(dp[j],dp[j-v]+w);
    }
    cout<<dp[V];
    return 0;
}
```



### 3.多重背包1.0

[多重背包]: https://www.acwing.com/problem/content/4/

```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    vector<int> dp(V+1);
    for(int i=0;i<N;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        
        for(int j=V;j>=0;j--)
        for(int k=1;k<=s && k*v<=j;k++)
        dp[j]=max(dp[j],dp[j-k*v]+k*w);
    }
    cout<<dp[V];
    return 0;
}
```

### 4.多重背包2.0(二进制优化)

[多重背包2.0]: https://www.acwing.com/problem/content/5/

```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    vector<pair<int,int>> goods;
    for(int i=0;i<N;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        for(int k=1;k<=s;k*=2)
        {
            s-=k;
            goods.push_back({v*k,w*k});
        }
        if(s>0)goods.push_back({v*s,w*s});
        
        
    }
    vector<int>dp(V+1);
    for(auto good:goods)
    {
        for(int j=V;j>=good.first;j--)
        {
            dp[j]=max(dp[j],dp[j-good.first]+good.second);
        }
    }
    cout<<dp[V];
    return 0;
}
```

### 5.混合背包

[混合背包]: https://www.acwing.com/problem/content/7/

超时版本：转化为01+完全，01反向，完全正向

```c++
#include<iostream>
#include<vector>

using namespace std;

struct good
{
    int kind;
    int v,w;
};
int main()
{
    int N,V;
    cin>>N>>V;
    vector<good> data;
    for(int i=0;i<N;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        if(s==-1)data.push_back({-1,v,w});
        else if(s==0)data.push_back({0,v,w});
        else if(s>0)
        for(int j=1;j<=s;j++)
        data.push_back({-1,v*j,w*j});
        
        
    }
    vector<int> dp(V+1);
    for(auto good:data)
    {

        if(good.kind==-1)
        {
            for(int j=V;j>=good.v;j--)
            dp[j]=max(dp[j],dp[j-good.v]+good.w);
        }
        else if(good.kind==0)
        {
            for(int j=good.v;j<=V;j++)
            dp[j]=max(dp[j],dp[j-good.v]+good.w);
        }
    }
    cout<<dp[V];
    return 0;
}
```

正确答案：通过二进制优化，直接转化为01

```c++
#include<iostream>
#include<vector>

using namespace std;


int main()
{
    int N,V;
    cin>>N>>V;
    vector<pair<int,int>> goods;
    for(int i=0;i<N;i++)
    {
        int v,w,s;
        cin>>v>>w>>s;
        if(s==-1)s=1;
        else if(s==0)s=V/v;
        
        for(int k=1;k<=s;k<<=2)
        {
            s-=k;
            goods.push_back({v*k,w*k});
        }
        if(s>0)
        {
            goods.push_back({s*v,s*w});
        }
    }
    vector<int> dp(V+1);
    for(auto good:goods)
    {
        for(int j=V;j>=good.first;j--)
        dp[j]=max(dp[j],dp[j-good.first]+good.second);
    }
    cout<<dp[V];
    return 0;
}
```

### 6.二维代价背包

[二维代价背包]: https://www.acwing.com/problem/content/8/

```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V,M;
    cin>>N>>V>>M;
    vector<vector<int>> dp(V+1,vector<int>(M+1,0));
    for(int i=0;i<N;i++)
    {
        int v,m,w;
        cin>>v>>m>>w;
        for(int j=V;j>=v;j--)
        for(int k=M;k>=m;k--)
        dp[j][k]=max(dp[j][k],dp[j-v][k-m]+w);
        
    }
    cout<<dp[V][M];
    return 0;
}
```

### 7.最优方案数

[最优方案数]: https://www.acwing.com/problem/content/11/

```c++
#include<iostream>
#include<vector>

using namespace std;
const int mod = 1e9 + 7;
int main()
{
    int N,V;
    cin>>N>>V;
    
    vector<int> dp(V+1,0);
    vector<int> cnt(V+1,1);
    
    for(int i=0;i<N;i++)
    {
        int v,w;
        cin>>v>>w;
        for(int j=V;j>=v;j--)
        {
            int tmp=dp[j-v]+w;
            if(tmp>dp[j])
            {
                cnt[j]=cnt[j-v];
                dp[j]=tmp;
            }
            else if(tmp==dp[j])
            {
                dp[j]=dp[j-v]+w;
                cnt[j]=(cnt[j]+cnt[j-v])%mod;
            }
        }
    }
    cout<<cnt[V];
    return 0;
}
```

### 8.最优方案

[最优方案]: https://www.acwing.com/problem/content/12/

```c++
#include<iostream>
#include<vector>

using namespace std;

int main()
{
    int N,V;
    cin>>N>>V;
    vector<int> v(N);
    vector<int> w(N);
    for(int i=0;i<N;i++)
    {
        cin>>v[i]>>w[i];
    }
    vector<vector<int>> dp(N+1,vector<int>(V+1,0));
    for(int i=N;i>0;i--)
    {
        for(int j=1;j<=V;j++)
        {
            if(j<v[i-1])
            dp[i][j]=dp[i-1][j];
            else
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-v[i-1]]+w[i-1]);
        }
    }
    vector<int> ans;
    int cur_v=V;
    for(int i=1;i<=N;i++)
    {
        if(i==N && cur_v>=v[i-1])
        {
            ans.push_back(i);
            break;
        }
        if(cur_v<=0)break;
        if(cur_v-v[i-1]>=0 && dp[i][cur_v]==dp[i+1][cur_v-v[i-1]]+w[i-1])
        {
            ans.push_back(i);
            cur_v-=v[i-1];
        }
        
    }
    for(int n:ans)
    cout<<n<<" ";

    return 0;
}
```

### 9.装入方案数

```c++
#include<iostream>
#include<vector>
#include<algorithm>
#include<fstream>
#include<string>
#include<numeric>
#include <sstream>
#include <iomanip>
#include<unordered_map>
#include<set>

using namespace std;
int main()
{
	int x, m;
	cin >> x >> m;

	vector<int> v(m);
	for (int i = 0; i < m; i++)cin >> v[i];
	int res = 0;
	vector<int> dp(x + 1,0);
	dp[0] = 1;
	//sort(v.begin(),v.end());
	for (auto t : v)
	{
		for (int j = x; j >= t; j--)
			dp[j] += dp[j - t];
	}
	cout << dp[x];
	return 0;
}
```



```java
int[][] dp[N+1][W+1]
dp[0][..] = 0
dp[..][0] = 0

for i in [1..N]:
    for w in [1..W]:
        dp[i][w] = max(
            把物品 i 装进背包,
            不把物品 i 装进背包
        )
return dp[N][W]
零钱问题2.0            
for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= amount; j++) {
        if (j - coins[i-1] >= 0)
            dp[i][j] = dp[i - 1][j] 
                     + dp[i][j-coins[i-1]];
return dp[N][W]
```

背包问题，i永远是物品价值，j永远是背包容量。塞得下时永远要考虑塞不下时的情况。

## 2.6 4键键盘，**另外一种状态转移不好找的时候考虑再设置一个变量进行遍历。**

[651.四键键盘（中等）](https://leetcode-cn.com/problems/4-keys-keyboard)

```c++
class Solution {
public:
    int maxA(int n) {
        if(n<=6)return n;

        vector<int> dp(n,0);
        for(int i=0;i<6;i++)dp[i]=i+1;
        for(int i=6;i<n;i++)
        {
            dp[i]=dp[i-1]+1;
            for(int j=3;j<i;j++)
            {
                dp[i]=max(dp[i],dp[j-2]*(i-j+1));
            }
        }
        return dp[n-1];
    }
};
```

**另外一种状态转移不好找的时候考虑再设置一个变量进行遍历。**

## 2.7股票问题（状态机）

```c++
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
              max(   选择 rest  ,             选择 sell      )

解释：今天我没有持有股票，有两种可能：
要么是我昨天就没有持有，然后今天选择 rest，所以我今天还是没有持有；
要么是我昨天持有股票，但是今天我 sell 了，所以我今天没有持有股票了。

dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
              max(   选择 rest  ,           选择 buy         )

解释：今天我持有着股票，有两种可能：
要么我昨天就持有着股票，然后今天选择 rest，所以我今天还持有着股票；
要么我昨天本没有持有，但今天我选择 buy，所以今天我就持有股票了。
```

有限交易范例

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n=prices.size();
        vector<vector<vector<int>>> dp(n,vector<vector<int>>(3,vector<int>(2,0)));
        dp[0][0][1]=-prices[0];
        dp[0][1][1]=-prices[0];
        dp[0][2][1]=-prices[0];
        // dp[0][1][1]=0;
        for(int i=1;i<n;++i)
        {
            for(int k=1;k<=2;++k)
            {
                dp[i][k][0]=max(dp[i-1][k][0],dp[i-1][k][1]+prices[i]);
                dp[i][k][1]=max(dp[i-1][k][1],dp[i-1][k-1][0]-prices[i]);
            }
        }
        return dp[n-1][2][0];
    }
};
```

小诀窍：

![](图片\状态机.png)

**画出状态图，几个箭头输入就代表变化时要比较几次。**

# 3.一些贪心问题

## 3.1调度区间

![](./图片/区间重叠问题.jpg)

```java
class Solution {
public:
static bool cmp(vector<int>& a, vector<int>& b)
{
    return a[1]<b[1]?true:false; 
}
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),cmp);
        int ans=0;
        int tail=intervals[0][1];
        for(int i=1;i<intervals.size();i++)
        {
            if(tail>intervals[i][0])ans++;
            else tail=intervals[i][1];
        }
        return ans;
    }
}
```

## 3.2最简覆盖一个区间的办法

```c++
leetcode剪视频
class Solution {
public:
static bool cmp(vector<int>& a, vector<int>& b)
{
    if(a[0]<b[0])return true;
    else if(a[0]==b[0])return a[1]>b[1]?true:false;
    else return false;
}
    int videoStitching(vector<vector<int>>& clips, int time) {
        sort(clips.begin(),clips.end(),cmp);
        int ans=0;
        int n=clips.size();
        int i=0;
        int curend,nextend;
        curend=nextend=0;
        while(i<n && clips[i][0]<=curend)
        {
            while(i<n && clips[i][0]<=curend)
            {
                nextend=max(clips[i][1],nextend);i++;
            }
            ans++;
            curend=nextend;
            if(curend>=time)return ans;
        }
        return -1;
        
    }
};
```

# 4.位操作

## 4.1冷知识

**利用异或操作** **`^`** **和空格进行英文字符大小写互换**

```c++
('d' ^ ' ') = 'D'
('D' ^ ' ') = 'd'
```

**判断两个数是否异号**

```c++
int x = -1, y = 2;
bool f = ((x ^ y) < 0); // true

int x = 3, y = 2;
bool f = ((x ^ y) < 0); // false
```

**不用临时变量交换两个数**

```c++
int a = 1, b = 2;
a ^= b;
b ^= a;
a ^= b;
// 现在 a = 2, b = 1
```

**加一减一**

```c++
int n = 1;
n = -~n;
// 现在 n = 2

int n = 2;
n = ~-n;
// 现在 n = 1
```

## 4.2常用技巧

### **4.2.1.消除数字 `n` 的二进制表示中的最后一个 1**

n&(n-1)

示范：

#### 汉明权重

```c++
int hammingWeight(uint32_t n) {
    int res = 0;
    while (n != 0) {
        n = n & (n - 1);
        res++;
    }
    return res;
}
```

#### 判断2的指数

一个数如果是 2 的指数，那么它的二进制表示一定只含有一个 1：

```c++
bool isPowerOfTwo(int n) {
    if (n <= 0) return false;
    return (n & (n - 1)) == 0;
}
```

### 4.2.2.查找只出现一次的元素

```c++
int singleNumber(vector<int>& nums) {
    int res = 0;
    for (int n : nums) {
        res ^= n;
    }
    return res;
}
```

# 5.数学类常见技巧

## 5.1 阶乘0位

```c++
int trailingZeroes(int n) {
    int res = 0;
    long divisor = 5;
    while (divisor <= n) {
        res += n / divisor;
        divisor *= 5;
    }
    return res;
}
```



# 5.2 找质数

```java
int countPrimes(int n) {
    boolean[] isPrim = new boolean[n];
    Arrays.fill(isPrim, true);
    for (int i = 2; i * i < n; i++) 
        if (isPrim[i]) 
            for (int j = i * i; j < n; j += i) 
                isPrim[j] = false;

    int count = 0;
    for (int i = 2; i < n; i++)
        if (isPrim[i]) count++;

    return count;
}
```

## 5.3 高效进行模幂运算

[372.超级次方（中等）](https://leetcode-cn.com/problems/super-pow)

```c++
int base = 1337;
// 计算 a 的 k 次方然后与 base 求模的结果
int mypow(int a, int k) {
    // 对因子求模
    a %= base;
    int res = 1;
    for (int _ = 0; _ < k; _++) {
        // 这里有乘法，是潜在的溢出点
        res *= a;
        // 对乘法结果求模
        res %= base;
    }
    return res;
}

int superPow(int a, vector<int>& b) {
    if (b.empty()) return 1;
    int last = b.back();
    b.pop_back();

    int part1 = mypow(a, last);
    int part2 = mypow(superPow(a, b), 10);
    // 每次乘法都要求模
    return (part1 * part2) % base;
}
```

高效乘方

```c++
int base = 1337;

int mypow(int a, int k) {
    if (k == 0) return 1;
    a %= base;

    if (k % 2 == 1) {
        // k 是奇数
        return (a * mypow(a, k - 1)) % base;
    } else {
        // k 是偶数
        int sub = mypow(a, k / 2);
        return (sub * sub) % base;
    }
}
```

## 5.4 缺失元素

1.位图思想

```c++
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        const int n = nums.size();
        vector<int> ans;
        bool flag[100001];
        memset(flag, false, sizeof(flag)); 
        for(int& num: nums)  //把有的数字都标记成true
            flag[num] = true;
        for(int i = 1; i <= n; ++i) //把没标记的加入答案
            if(!flag[i])
                ans.push_back(i);
        return ans;
    }
};

作者：MGA_Bronya
链接：https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/solution/448-zhao-dao-suo-you-shu-zu-zhong-xiao-s-pg4i/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

2.求和相减

## 5.5水塘抽样

```c++
int getRandom() {
        int k = 2;
        int res = head->val;
        ListNode* curr = head->next;
        while (curr != nullptr)
        {
            if (rand() % k == 0)
            {
                res = curr->val;
            }
            curr = curr->next;
            ++k;
        }

        return res;
    }

作者：ffreturn
链接：https://leetcode-cn.com/problems/linked-list-random-node/solution/382-cjian-dan-yi-dong-de-xu-shui-chi-jie-222u/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

## 5.6 链表去重

```c++
ListNode deleteDuplicates(ListNode head) {
    if (head == null) return null;
    ListNode slow = head, fast = head;
    while (fast != null) {
        if (fast.val != slow.val) {
            // nums[slow] = nums[fast];
            slow.next = fast;
            // slow++;
            slow = slow.next;
        }
        // fast++
        fast = fast.next;
    }
    // 断开与后面重复元素的连接
    slow.next = null;
    return head;
}

int removeElement(int[] nums, int val) {
    int fast = 0, slow = 0;
    while (fast < nums.length) {
        if (nums[fast] != val) {
            nums[slow] = nums[fast];
            slow++;
        }
        fast++;
    }
    return slow;
}
```

## 5.7 移动0和去重（双指针）

```c++
int removeElement(int[] nums, int val) {
    int fast = 0, slow = 0;
    while (fast < nums.length) {
        if (nums[fast] != val) {
            nums[slow] = nums[fast];
            slow++;
        }
        fast++;
    }
    return slow;
}
```

## 5.8 快速区间求和（前缀和）

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int,int> m;
        m[0]=1;
        int ans=0;
        int sum=0;
        for(int& n:nums)
        {
            sum+=n;
            //if(sum>k)
            if(m.count(sum-k))
            ans+=m[sum-k];
            m[sum]++;
        }
        return ans;
    }
};
```

## 5.9 区间快速加和（差分区间）

```c++
class Solution {
public:
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> diff(n,0);
        for(auto ve:bookings)
        {
            diff[ve[0]-1]+=ve[2];
            if(ve[1]<diff.size())
            diff[ve[1]]-=ve[2];
        }
        vector<int> ans(n,0);
        ans[0]=diff[0];
        for(int i=1;i<diff.size();i++)
        {
            ans[i]=diff[i]+ans[i-1];
        }
        return ans;
    }
};
```



# 5.小技巧

### 二维偏序问题先升序后降序

### stl找最大max——element

### stl求和accumulate(vec.begin() , vec.end() , 42);

### stl sort(vec.begin(),vec.end(),grerater<int>())

### 整数个数time += (p - 1) / K + 1;

### 三维vector初始化

vector<vector<vector<int> > > vecInt(m, vector<vector<int> >(n, vector<int>(l)));

### 小顶堆声明

priority_queue <int,vector<int>,greater<int> > q;

## 5.1输入输出

```c++
//2 3 10
//5 3
//9 7

//使用stringstream
	int n, m, k;
	cin >> n >> m >> k;
	if (cin.get() == '\n')
	cin.unget();
	string line;
	vector<vector<int>> mat;
	for(int i=0;i<n;i++)
	{
		getline(cin, line);
		stringstream ss(line);
		vector<int> tmp;
		string token;
		while (ss >> token)
		{
			tmp.push_back(stoi(token));
		}
		mat.push_back(tmp);

	}
//使用cin
	int n, m, k;
	cin >> n >> m >> k;
	vector<vector<int>> mat;
	for(int i=0;i<n;i++)
	{
		vector<int> tmp(m);
		for(int i=0;i<m;i++)cin>>tmp[i]
		mat.push_back(tmp);

	}
//scanf
	int m;//行
	int n;//每行的数字个数

	vector<vector<int> > vec;
	scanf_s("%d,%d", &m, &n, 1, 1);
	//scanf("%d,%d", &n,&m);
	for (int i = 0; i < m; ++i){
		for (int j = 0; j < n; ++j)
        {
            vector<int> vectmp;
			int tmp;
			scanf_s("%d,", &tmp, 1);
			//scanf("%d,", &tmp);
			vectmp.push_back(tmp);
		}
		vec.push_back(vectmp);
	}
————————————————
版权声明：本文为CSDN博主「qq_40602964」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40602964/article/details/95252976

```

```c++
//1,2,3:3
	int pos = s.find_first_of(":");
	string arrStr = s.substr(0, pos);
	int k = atoi(s.substr(pos + 1).c_str());
	vector<string> ans;
	string line,token;
	cin >> line;
	stringstream ss(line);
	while (getline(ss, token, ','))
	{
		ans.push_back(token);
	}	
//1,2,3
//getline
	vector<string> ans;
	string line,token;
	cin >> line;
	stringstream ss(line);
	while (getline(ss, token, ','))
	{
		ans.push_back(token);
	}	
//scanf
	int n;
	vector<int> vec;
	scanf_s("%d", &n, 1);//读入n,不需要逗号
	//scanf("%d", &n);
	for (int i = 0; i < n; ++i){
		int tmp;
		scanf_s("%d,", &tmp, 1);//分别读入n个数，'%d'后面加','
		//scanf("%d,", &tmp);//分别读入n个数，'%d'后面加','
		vec.push_back(tmp);
	}
————————————————
版权声明：本文为CSDN博主「qq_40602964」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40602964/article/details/95252976
```

```c++
//未告诉输入个数，以回车结束
	vector<int> vec;
	int tmp;
	char ch = 'a';
	 while(ch != '\n')
	{
		scanf_s("%d", &tmp, 1);//数字
		//cout << "get_tmp:" << tmp << " ";
		ch = getchar();	//空格或逗号
		//cout << "get_ch:" << ch << endl;
		vec.push_back(tmp);
	} 

	vector<int> ans;
	string line, token;
	getline(cin, line);
	stringstream ss(line);
	while (getline(ss, token,' '))
	{
		ans.push_back(stoi(token));
	}

————————————————
版权声明：本文为CSDN博主「qq_40602964」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40602964/article/details/95252976
```

