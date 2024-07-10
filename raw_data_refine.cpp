#include<bits/stdc++.h>
using namespace std;
int main(){
	freopen("array.txt","r",stdin);
	freopen("raw_data1.txt","w",stdout);
	string s;
	while(cin>>s){
		for(register int i=0;i<s.length();i++){
			if(s[i]>='0' and s[i]<='9')
				cout<<s[i];
		}
		cout<<endl;
	}
	return 0;
}