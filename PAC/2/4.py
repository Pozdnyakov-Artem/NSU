dic={'1':'odin','2':'dva','3':'tri'}

st="1 5 4 2 8 9 2 3"

m=st.split()
for ind,word in enumerate(m):
    word=word.strip(' ,./?\\"\'*&(){}[]%^$#@!')
    if word in dic.keys() or word.lower() in dic.keys():
        m[ind]=dic[word]

print(' '.join(m))