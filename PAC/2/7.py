def geom(start,q):
    fst=start
    while True:
        yield fst
        fst*=q


length = 5
shag = geom(1,5)

for i in range(length):
    print(next(shag))