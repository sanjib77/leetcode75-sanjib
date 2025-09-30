class ReverseVowelsOfaString:
    def reverseVowels(self, s: str) -> str:
        print(list(s))
        l = list(s) 
        # ['I', 'c', 'e', 'C', 'r', 'e', 'A', 'm']
        vowels = set('aeiouAEIOU')
        i, j = 0, len(l) - 1
        while i < j:
            if l[i] not in vowels:
                i += 1
            elif l[j] not in vowels:
                j -= 1
            else:
                l[i], l[j] = l[j], l[i]
                i += 1
                j -= 1
        return ''.join(l)

a= ReverseVowelsOfaString().reverseVowels("IceCreAm")
print(a)


