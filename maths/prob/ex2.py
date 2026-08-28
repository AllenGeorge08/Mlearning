import pandas as pd 

die1 = [0.10, 0.20, 0.15, 0.25, 0.20, 0.10]

die2 = [0.05, 0.15, 0.20, 0.30, 0.20, 0.10]

outcomes = [i+1 for i in range(len(die1))]
print(f"Outcomes : {outcomes}")

joint = []
for px in die1:
    row = []
    for py in die2:
        row.append(px*py)
    joint.append(row)

i = 0
for row in joint:
    print(f"Row No: {i} : {row}")
    i+=1


marginal_x =[sum(row) for row in joint]


marginal_y = [sum(joint[row][col] for row in range(6)) for col in range(6)]

print("Marginal X: ",marginal_x)
print("Marginal Y: ",marginal_y)
print("-"*50)

independent = True

for x in range(len(die1)):
    for y in range(len(die2)):
        if joint[x][y] != die1[x]*die2[y]:
            print(joint[x][y])
            independent = False 
    
    
# Labelled rows and columns
df = pd.DataFrame(joint,index=outcomes,columns=outcomes)
df.index.name = 'Die 1'
df.columns.name = 'Die 2'
print(df)

print(f"Are the both dies' independent : {independent} ")