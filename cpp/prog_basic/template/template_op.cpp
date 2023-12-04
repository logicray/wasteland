#include<iostream>

template<typename T>
T max(T a, T b, T c){
	a = a > b ? a:b;
	a = a > c ? a:c;
	return a;
}

template<class numtype>
class Compare{
  private:
    numtype x,y;
  public:
    Compare(numtype a, numtype b);//{ x = a; y = b;}
    numtype max();//{return x>y?x:y;}
    numtype min();//{return x<y?x:y;}
};

template<class numtype>
Compare<numtype>::Compare(numtype a, numtype b){
  x = a;
  y = b;
}

template<class numtype>
numtype Compare<numtype>::max(){
  return x > y ? x:y;
}

template<class numtype>
numtype Compare<numtype>::min(){
 return x < y ? x :y;
}
     

int main(int argc, char* argv[]){
	std::cout << max(3, 9, 10) << std::endl;
    std::cout << "split line" << std::endl;
    Compare<int> com1(4, 12);
    std::cout << com1.max() << ", " << com1.min() << std::endl;
    return 0;
}
