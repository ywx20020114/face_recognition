
# 返回预测出来的最有可能的姓名
# 返回值 ： 姓名
def get_name():
    f = open('apps/Name/temp.txt', encoding='gbk')
    txt = []
    for line in f:
        txt.append(line.strip())
    name_str = txt[0]
    return name_str


