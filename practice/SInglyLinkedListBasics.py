class Node:
    def __init__(self, data):
        """Initialize a new node with given data.
        
        Args:
            data: The data to be stored in the node
        """
        self.data = data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None
    
    def traverse(self):
        """Traverse and print the linked list.
        
        Returns:
            str: String representation of the linked list
        """
        result = []
        curr_node = self.head
        while curr_node:
            result.append(str(curr_node.data))
            curr_node = curr_node.next
        return " -> ".join(result + ["None"])

    def delete_node(self, node_to_delete):
        """Delete a specific node from the linked list.
        
        Args:
            node_to_delete: The node to be deleted
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if self.head is None:
            return False
            
        if self.head == node_to_delete:
            self.head = self.head.next
            node_to_delete = None
            return True
            
        curr_node = self.head
        while curr_node.next:
            if curr_node.next == node_to_delete:
                curr_node.next = curr_node.next.next
                node_to_delete = None
                return True
            curr_node = curr_node.next
        return False
#Example usage: 
if __name__ == "__main__":
    linked_list = SinglyLinkedList()
    linked_list.head = Node(1)
    second_node = Node(2)
    third_node = Node(3)
    
    linked_list.head.next = second_node
    second_node.next = third_node
    
    print("Original List:")
    print(linked_list.traverse())
    
    linked_list.delete_node(second_node)
    
    print("After Deleting Node with value 2:")
    print(linked_list.traverse())
    print(second_node.data) # Should print 2, but node is deleted from list
""" The reasoning behind printing second_node.data is to show that 
the node object still exists in memory even after being removed from the linked list. 
The deletion operation only removes the reference to the node from the linked list, 
but does not delete the node object itself. Thus, we can still access its data attribute. """