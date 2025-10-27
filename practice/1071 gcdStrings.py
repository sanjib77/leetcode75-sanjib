class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        # Step 1: Check if str1 and str2 are compatible
        # If str1 + str2 != str2 + str1, then they are built from different patterns
        # So no common base string (GCD string) exists
        if str1 + str2 != str2 + str1:
            return ""

        # Step 2: Compute GCD of lengths manually using Euclidean algorithm
        # This gives us the length of the longest possible repeating unit
        def compute_gcd(a, b):
            while b != 0:
                a, b = b, a % b
            return a

        # Step 3: Use GCD length to extract candidate substring
        # This substring is the potential base pattern that could repeat to form both strings
        gcd_len = compute_gcd(len(str1), len(str2))
        candidate = str1[:gcd_len]

        # Step 4: Return the candidate if both strings are compatible (already checked)
        return candidate
sol = Solution()
result = sol.gcdOfStrings("ABABAB", "ABAB")
print("Final Output:", result)