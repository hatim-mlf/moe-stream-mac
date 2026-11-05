int median5_i32(int a,int b,int c,int d,int e) {
    int v[5]={a,b,c,d,e};
    for(int i=0;i<5;i++) for(int j=i+1;j<5;j++) if(v[i]>v[j]){int t=v[i];v[i]=v[j];v[j]=t;}
    return v[2]+1;
}
