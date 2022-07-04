# 关于ACM模式：C++和Python的输入输出

大部分企业校招似乎都用牛客网，最近接触了ACM模式，原来就是OJ的输入输出，因为最近都是在leetcode上练题，早就不怎么熟悉OJ的输入和输出了，所以这里总结一下需要注意的地方。

一个练习场：[OJ在线编程常见输入输出练习场 ](https://ac.nowcoder.com/acm/contest/5657#question)

因为我常用的语言就是C++和Python，而ACM模式要适应的就是输入s输出，所以我分别总结一下这两个语言的输入输出要注意的地方：

## C++

第一个让我比较犹豫的是A+B(7)这个输入，因为需要判断行，而据我所知cin应该是不读入'\n'的。

**Takeaways:**

1. 要输⼊⼀⾏的数据的话：
   如果是==string  s== ，则⽤ `getline(cin,s)`，需要注意的是前面如果有换行的输入（比如c`cin>>n`后再getline），一定要在前面加上`getchar();`（用来读取空格），否则会直接只读入要读的字符串前面的`\n`;  在头⽂件 `#include  <string>` ⾥⾯；
   如果是 ==char  str[100]== , 则⽤ `cin.getline(str,  100)`;  在头⽂件 `#include  <iostream>` ⾥⾯，也可以⽤ `gets(str);1`

2. 想要读入空格和回车：用`scanf("%c",&ch);`

3. 读入一行数据后，怎么分隔用空格或者','等隔离的数据？用stringstream！

	a. 求一行被空格隔开的整数的和：

  ```c++
  #include <iostream>
  #include <string>
  #include <stdio.h>
  #include <sstream>
  
  using namespace std;
  
  // 求一行被空格隔开的整数的和
  void Sum() {
  	string line;
  	cout << "输入一行数据： " << endl;
  
  	getline(cin, line);	//getline函数将输入的一行数据用字符串line接收
  
  	int sum = 0, x;
  	stringstream ss(line);//将字符串line放入到输入输出流ss中
  	cout <<"输入输出流中的数据： "<< ss.str() << endl;
  
  	while (ss >> x) sum += x;//求和
  
  	cout <<"和为： "<< sum << endl;
  }
  
  int main() {
  	Sum();
  	return 0;
  }   
  ```
  
  b. 将一行以逗号分隔的整数存放到数组中:
  
  ```C++
  #include <iostream>
  #include <string>
  #include <stdio.h>
  	#include <sstream>
  #include <vector>
  
  using namespace std;
  
  // 将一行以逗号分割的整数存放到数组中
  void test() {
  	vector<int> ve;
  	string word;
  	//因为输入是一行以逗号分割的字符串，所以这里也可以使用cin
  	//cin >> word;//输入：1,2,3,4,5,6,7,8,9
  	getline(cin, word);//输入一行字符串，用word接收
  	cout << "word: " << word << endl;
  	string str;
  	stringstream ss(word);//将输入的字符串放入到流ss中
  	while (getline(ss, str, ',')) {//使用getline函数从流中读取字符串到str中，并以','作为分割，即遇到一个','函数返回
  		ve.push_back(stoi(str));	//将读取到的字符串先进行类型转换，再添加到数组中
  	}
  
  	// 清空 ss
  	ss.clear();
  	cout << "打印数组: " << endl;
  	for (auto x : ve) {
  		cout << x << " ";//1 2 3 4 5 6 7 8 9
  	}
  	cout << endl;
  }
  
  int main() {
  	test();
  	return 0;
  }
  ```



- `cin>>`：

  遇“空格”、“TAB”、“回车”就结束

- `cin.getline()`:

  可以接收空格，默认换行'\n'或'\0'结束。也可以自己设置：

  `cin.getline(字符指针(char*), 字符个数N(int), 结束符(char))`

- `cin.get()`:

  获取字符，包括空格和'\0'。可以指定长度：`cin.get(ch,len)`

- `getline(cin,s)`:

  需要包括<string>，默认'\n'或者'\0'结束。可以指定：

  `geline(cin,str,结束符(char))`

- `getchar()`:

  获取一个字符

- string转char*：`c_str()`

- char*转string：`string()`

- string转int: `stoi()`

## Python

Python的输入和格式化输出我平时用的比较少，所以这里把很基础的也写下：

**Takeaways:**

1. `input()`:

   Python 3.x中input读入的是字符串，必须进行转换才能变为数值。比如：`num=int(input())`

2. 读入一行：

   用`a, b, c = map(int, input().split())`

3. 格式化输出：

   可以选择与C语言相似的：`print('常量pi的值近似是： %4.2f。' %math.pi)`