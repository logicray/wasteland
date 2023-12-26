/**
 * @author logic-pw
 * @date 2023.12.01
 */

#include<iostream>
#include<cstring>


using namespace std;

bool is_palindrome(char*, int, int);
int longest_palindrome(char*);
int longest_palindrome2(char*);
int longest_palindrome3(char*);
int longest_palindrome4(char*);

int main(int argc, char* argv[]){
	// 0 是 false，所有的非0都是true
	char s[30]; //注意最后有个终止符号
	cin >> s;
	cout << "input str len: " << strlen(s) << endl;
    //char t[3] = "ss";
	//bool res = is_palindrome(s, 3, 5);
	//cout << res << endl;
	int res2 = longest_palindrome3(s);
	cout << res2 << endl;
	return 0;
}

/*
 * use brute-froce 
 * O(n^3)
 */
int longest_palindrome(char* str){
    int l = strlen(str);
	int max_len = 0;
	for(int i =0; i < l-1; i++){
		for(int j = i+1; j< l; j++){
			cout << i << " " << j+1 << endl;
			if(is_palindrome(str, i, j+1)){
                if(j+1-i > max_len){
					max_len = j+1-i;
				}
			}
		}
	}
	return max_len;
}

/**
 * s include, e not include
 */
bool is_palindrome(char* str, int s, int e){
	for(int i=s; i<e; i++){
        int j = e - i + s - 1;
		if(str[i] != str[j]){
			return false;
		}
	}
	return true;
}

/**
 * 遍历，中心扩散法
 */
int longest_palindrome2(char* str){
	int len = strlen(str);
	int max_len = 0;
	for(int i=0; i< len-1; i++){
		int left = i, right = i;
		while (right < len && str[right] == str[right+1]){
			right++;
		}
        while(left-1 >= 0 && right+1 <= len){
			if (str[left-1] != str[right+1]){
				break;
			}
			left--;
			right++;
		}
		cout << i << ":" <<right - left + 1 << endl;
		if(right - left + 1 > max_len){
            max_len = right - left + 1;
		}
	}
	return max_len;
}

/**
 * 动态规划
 */
int longest_palindrome3(char* str){
   const int len = strlen(str);
   int max_len = 0;
   bool dp[len][len];// = new bool[len][len];
   for (int j=0; j<len; j++){
	   for(int i=0; i<=j; i++){
		   if (str[i] != str[j])
			   continue;
		   if (i == j)
			   dp[i][j] = true;
		   else if (j - i <= 2)
			   dp[i][j] = true;
           else{
			   dp[i][j] = dp[i+1][j-1];
		   }
		  max_len = j-i+1 > max_len ? j-i+1:max_len;
	   }
   }
   return max_len;
}

/**
 * manacher
 */
int longest_palindrome4(char* str){
   int max_len = 0;
   return max_len;
}
