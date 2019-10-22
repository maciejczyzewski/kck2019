import os

for idx in range(0, 20 + 1):
    name = f"samolot{str(idx).zfill(2)}.jpg"
    cmd = f"curl https://www.cs.put.poznan.pl/mtomczyk/kck/Lab4_images/planes/{name} > {name}"
    print(name)
    print(cmd)
    os.system(cmd)
