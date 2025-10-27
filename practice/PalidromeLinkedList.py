# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

node1 = ListNode(7)
node2 = ListNode(11)
node3 = ListNode(3)
node4 = ListNode(2)
node5 = ListNode(9)

node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
