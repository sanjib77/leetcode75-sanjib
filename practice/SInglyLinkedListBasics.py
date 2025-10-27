class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def traverse(head):
    currNode = head
    while currNode:
        print(currNode.data, end = "-> ")
        currNode = currNode.next
    print("Null")

node1 = Node(7)
node2 = Node(11)
node3 = Node(3)
node4 = Node(2)
node5 = Node(9)

node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5

traverse(node1)

#7-> 11-> 3-> 2-> 9-> Null
#now we will try to delete a node in singly linked list
def deleteNode(head,nodeTodelete):
    if head is None:
        print("Empty linked list")
        return
    currNode = head
    while currNode:            
        if currNode.next == nodeTodelete:
            currNode.next = currNode.next.next
            # del nodeTodelete
            nodeTodelete = None

            return head
        currNode = currNode.next
    return head


#now we will delete 3rd node which has value 3
head = node1
print(head.data)
deleteNode(head,node3)

print(head.data)
traverse(head)
#7-> 11-> 2-> 9-> Null

print(node3)
