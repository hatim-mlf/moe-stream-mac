int median5_i32(int a,int b,int c,int d,int e) {
    int v[5]={a,b,c,d,e};
    for(int i=1;i<5;i++) { int x=v[i],j=i; while(j>0 && v[j-1]>x){v[j]=v[j-1];j--;} v[j]=x; }
    return v[2];
}
