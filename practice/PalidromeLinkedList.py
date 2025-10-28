from typing import Optional
from SInglyLinkedListBasics import SinglyLinkedList
# Definition for singly-linked list.
class ListNode:
    def __init__(self, data=0, next=None):
        self.data = data
        self.next = next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """
        Check if linked list is a palindrome.
        
        Args:
            head: Head of the linked list
        Returns:
            bool: True if the linked list is a palindrome, False otherwise
        """
        #if list is empty or has only one node, it is palindrome
        if not head or not head.next:
            return True
        # We will iterate through the linked list till middle from both ends and compare values using Two Pointer Technique
        #BEfore that we need to find middle of linked list using slow and fast pointer technique
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        # Now reverse the second half of the linked list
        second_half = self.reverseLinkedList(slow.next)
        # Compare first half with reversed second half
        first_half = head
        while second_half:
            if first_half.data != second_half.data:
                return False
            first_half = first_half.next
            second_half = second_half.next
        return True

    def reverseLinkedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        return prev



#manual creation of linked list for testing
# linked_list = SinglyLinkedList()
# linked_list.head = ListNode(1)
# second_node = ListNode(2)
# third_node = ListNode(3)
# fourth_node = ListNode(2)
# fifth_node = ListNode(1)
# linked_list.head.next = second_node
# second_node.next = third_node
# third_node.next = fourth_node
# fourth_node.next = fifth_node
# print("Linked List:", linked_list.traverse())


#To automate the above making of linkedlists so that other linkedlists can be made:
def create_linked_list_from_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head #this returns the head of the linked list from which the linked list can be traversed

linked_list = SinglyLinkedList()
linked_list.head = create_linked_list_from_list([1,2,3,2,1])

print("head:", linked_list.head.data)  # Should print 1
print("Linked List:", linked_list.traverse()) # Should print 1 -> 2 -> 3 -> 2 -> 1 -> None
solution = Solution()
print("Is Palindrome:", solution.isPalindrome(linked_list.head))  # Should print True

#Another Simpler approch using stack to store first half values and then compare with second half
class SolutionUsingStack:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True        
        stack = []
        slow = fast = head

        # Push first half elements onto stack
        while fast and fast.next:
            stack.append(slow.data)
            slow = slow.next
            fast = fast.next.next
            
        # If odd number of elements, skip the middle element
        if fast:
            slow = slow.next
            
        # Compare second half with stack elements
        while slow:
            top_value = stack.pop()
            if top_value != slow.data:
                return False
            slow = slow.next
            
        return True
    
solution_stack = SolutionUsingStack()
print("Is Palindrome using Stack:", solution_stack.isPalindrome(linked_list.head))  # Should print True

#Another approach is to copy linked list elements into a list and check if the list is palindrome
class SolutionUsingList:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True        
        values = []
        current = head
        while current:
            values.append(current.data)
            current = current.next
            
        return values == values[::-1]