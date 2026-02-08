st=input()

st=st.replace(' ','')

print("Pal" if st==st[::-1] else "Not pal")