import random


class Node:

    def __init__(self, lower, higher, b, u=None, mean=None, count=0, active=False):

        self.left = None
        self.right = None
        self.lower, self.higher = lower, higher
        self.b = b
        self.u = u
        self.mean = mean
        self.count = count
        self.active = active

    # Print the tree
    def print_tree(self):
        if self.left and self.left.active:
            self.left.print_tree()
        print(f"({self.lower}, {self.higher}), b: {self.b}, u: {self.u}, mean {self.mean}, times {self.count}"),
        if self.right and self.right.active:
            self.right.print_tree()

    def print_tree_depth(self, depth):
        root = self
        if root:
            print(f"({self.lower}, {self.higher}), b: {self.b}, u: {self.u}, mean {self.mean}, times {self.count}")
            if root.left and height(root.left) <= depth:
                root.left.print_tree_depth(depth - 1)
            if root.right and height(root.right) <= depth:
                root.right.print_tree_depth(depth - 1)

    def in_order_traversal(self, root):
        res = []
        if root:
            res = self.in_order_traversal(root.left)
            res.append(root)
            res = res + self.in_order_traversal(root.right)
        return res

    def path(self, root):
        node = self
        path = [root]
        while root != node:
            if node.higher <= root.left.higher:
                root = root.left
                path.append(root)
            elif node.lower >= root.right.lower:
                root = root.right
                path.append(root)
        return path

    def duplicate_tree(self):
        old_root = self
        new_root = Node(old_root.lower, old_root.higher, old_root.b, old_root.u, old_root.mean, old_root.count,
                        old_root.active)
        if old_root.left:
            new_root.left = old_root.left.duplicate_tree()
        if old_root.right:
            new_root.right = old_root.right.duplicate_tree()
        return new_root
        # for node in self.in_order_traversal(root):

    def first_leaf(self):
        node = self
        if node.left and node.active:
            node = node.first_leaf()
        elif node.right and node.active:
            node = node.first_leaf()
        return node

    def first_leaf_and_parent(self):
        parent = self
        leaf = self
        if parent.left and parent.left.active:
            leaf = parent.left
        elif parent.right and parent.right.active:
            leaf = parent.right
        if (leaf.left and leaf.left.active) or (leaf.right and leaf.right.active):
            parent, leaf = leaf.first_leaf_and_parent()
        return parent, leaf

    def remove(self, node):
        if self.left == node:
            self.left = None
        elif self.right == node:
            self.right = None
        else:
            raise Exception('Node not found')

    def pre_order_trav(self):
        res = []
        node = self
        st = []
        while node or st:
            while node.active:
                res.append(node)
                st.append(node)
                node = node.left
            temp = st[-1]
            st.pop()
            if temp.right:
                node = temp.right
        return res

    def pre_order_traversal(self):
        res = []
        st = []
        node = self
        while node or st:
            while node:
                if node.active:
                    res.append(node)
                st.append(node)
                node = node.left
            temp = st[-1]
            st.pop()
            if temp.right:
                node = temp.right
        return res

    def find(self, node):
        root = self
        while root.lower != node.lower or root.higher != node.higher:
            if root.left:
                if root.left.higher >= node.higher:
                    root = root.left
            if root.right:
                if root.right.lower <= node.lower:
                    root = root.right
        return root


def height(node):
    if node is None:
        return 0
    else:
        return max(height(node.left), height(node.right)) + 1


