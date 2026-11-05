int is_ascii_identifier(const char*t){if(!t||!*t)return 0;for(size_t i=0;t[i];i++){char c=t[i];if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'))return 0;}return 1;}
