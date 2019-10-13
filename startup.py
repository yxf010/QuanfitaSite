#!/usr/bin/env python
import sys
import time
import markdown
import codecs
import glob
import os
import sqlite3

css = '''
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<style type="text/css">
</style>
'''
"""
def run():
    print('start up')
    conn = sqlite3.connect('db.sqlite3')
    con = conn.cursor()
    print(os.path.dirname(os.path.dirname(__file__))+'/templates/blogs/*.md')
    for filename in glob.glob(os.path.dirname(__file__)+'/templates/blogs/*.md'):
        print("do:"+filename)
        with codecs.open(filename, mode="r", encoding="utf-8") as input_file:
            content = input_file.read()
            filelines = content.splitlines()
            dic = {}
            i = 1
            if filelines[0] == '---':
                while filelines[i] != '---':
                    if filelines[i] == '---':
                        break
                    else:
                        try:
                            t,c = filelines[i].split(': ',1)
                        except:
                            t = filelines[i].split(': ',1)[0]
                            c = None
                        if not c:
                            c = []
                            j = 1
                            while filelines[i+j][0] == '-':
                                c.append(filelines[i+j].split('- ',1)[-1])
                                j += 1
                            i = i+j - 1
                        dic.update({t:c})
                    i += 1
            mark = '\n'.join(filelines[i+1:])
            content = markdown.markdown(mark,extensions=[
                                                # 包含 缩写、表格等常用扩展
                                                'markdown.extensions.extra',
                                                # 语法高亮扩展
                                                'markdown.extensions.codehilite',
                                                #允许我们自动生成目录
                                                 'markdown.extensions.toc',
                                                ])
            name = filename.split('/')[-1].split('.')[0].replace('_',' ')
            create_time = modify_time = time.localtime(time.time())
            try:
                category = con.execute("SELECT id FROM Blog_category WHERE name=='{}'".format(dic['category']))
            except KeyError:
                dic.update({'category':'test'})
                category = con.execute("SELECT id FROM Blog_category WHERE name=='{}'".format(dic['category']))
            except:
                con.execute("INSERT INTO Blog_category VALUES(?,?)" % (None, dic['category']))
                con.commit()
                category = con.execute("SELECT id FROM Blog_category WHERE name=='{}'".format(dic['category']))
            try:
                tag = con.execute("SELECT id FROM Blog_tag WHERE name==\"{}\"".format(dic['tags']))
            except KeyError:
                dic.update({'tags':'test'})
                tag = con.execute("SELECT id FROM Blog_tag WHERE name=='{}'".format(dic['tags']))
            except:
                con.execute("INSERT INTO Blog_tag VALUES({},{})".format(1,dic['tags']))
                con.commit()
                tag = con.execute("SELECT id FROM Blog_tag WHERE name=='{}'".format(dic['tags']))
            print("INSERT INTO Blog_blog(body,created_time,modified_time,category,tags) VALUES (?,?,?,?,?,?)",(content,create_time,modify_time,category,tag))
            tmp = con.execute("SELECT * FROM Blog_blog")
            print([i for i in tmp])
            con.execute("INSERT INTO Blog_blog VALUES ({},\
                '''{}''',\
                \"{}\",\
                \"{}\",\
                {},{},\
                {},\
                \"{}\")".format(1,content,create_time,modify_time,None,category,tag,0))
            con.commit()
    conn.close()
    return css+res
"""
def main(name):
    filename = os.path.dirname(__file__)+'/media/blogs/{}.md'.format(name)
    with codecs.open(filename, mode="r", encoding="utf-8") as input_file:
        content = input_file.read()
        filelines = content.splitlines()
        dic = {}
        i = 1
        if filelines[0] == '---':
            while filelines[i] != '---':
                if filelines[i] == '---':
                    break
                else:
                    try:
                        t,c = filelines[i].split(': ',1)
                    except:
                        t = filelines[i].split(': ',1)[0]
                        c = None
                    if not c:
                        c = []
                        j = 1
                        while filelines[i+j][0] == '-':
                            c.append(filelines[i+j].split('- ',1)[-1])
                            j += 1
                        i = i+j - 1
                    dic.update({t:c})
                i += 1
        mark = '\n'.join(filelines[i+1:])
        content = markdown.markdown(mark,extensions=[
                                            # 包含 缩写、表格等常用扩展
                                            'markdown.extensions.extra',
                                            # 语法高亮扩展
                                            'markdown.extensions.codehilite',
                                            #允许我们自动生成目录
                                             'markdown.extensions.toc',
                                            ])
        print(css+"\n"+content)

if __name__ == '__main__':
    #Tensorflow实现Neural Style图像风格转移
    main('Windows10+Anaconda+TensorFlow(CPU & GPU)环境快速搭建')
