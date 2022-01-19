import sys 

ans = {}
with open(sys.argv[1], 'r') as fp:
    fp.readline()
    for x in fp:
        _, name, label = x.strip().split(',')
        ans[name] = label
correct = 0 
with open(sys.argv[2], 'r') as fp:
    fp.readline()
    for x in fp:
        _, name, label = x.strip().split(',')
        if ans[name] == label:
            correct += 1

print(f'Accuracy={correct/len(ans)}')
with open(sys.argv[3], 'w') as fp:
    fp.write(f'Accuracy={correct/len(ans)}')