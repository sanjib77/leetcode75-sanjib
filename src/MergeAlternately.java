public class MergeAlternately {
    public String mergeAlternately(String word1, String word2) {
       int l1 = word1.length();
       int l2= word2.length();
        int d=Math.min(l1,l2);
        String s="";
        for(int i=0; i<d; i++){
            s = s + word1.charAt(i) + word2.charAt(i);
            if (i==d-1 && l1!=l2){
                if (l1>l2){
                    s=s+word1.substring(d);
                } else {
                    s=s+word2.substring(d);
                }
            }
        }
        return s;
    }
}