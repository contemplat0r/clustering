

codes_file = open('codes.txt')

codes_lines = codes_file.readlines()

codes_file.close()
'''
for line in codes_lines:
    print(line)
'''


step = 3
print(len(codes_lines))
print(len(codes_lines) / 3)

name_code_lines = []
for i in range(3, len(codes_lines), 3):
    #print(i, codes_lines[i - 1].strip(), codes_lines[i].strip())
    name_code_lines.append(';'.join((codes_lines[i - 1].strip(), codes_lines[i].strip())))

for line in name_code_lines:
    print(line)

codes_csv_file = open('codes.csv', 'w')
codes_csv_file.write('\n'.join(name_code_lines))
codes_csv_file.flush()
codes_csv_file.close()
