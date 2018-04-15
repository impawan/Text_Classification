# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
from lxml import etree
import time

parser = etree.HTMLParser()
report_name = "assig_desc.csv"
FileList="file_list.txt"
def indirect_load(FileList):
    file_names  = open(FileList,'r')
    for file in file_names:
        links = open(file.rstrip(),'r')
        for link in links:
            create_report(link)

def create_report(url):
    while True:
        try : 
            response = requests.get(url) 
            break
        except requests.exceptions.RequestException:
            time.sleep(300)
    print (url)
    tree = etree.fromstring(response.text, parser)
    assignee = ''.join(tree.xpath("//*[@id='bz_show_bug_column_1']/table/tr[13]/td/span/span/text()")).strip()
    summary = ''.join(tree.xpath("//*[@id='c0']/pre/text()")).rstrip()
    summary = summary.replace("\r","")
    summary = summary.replace("\n","")
    summary = str(summary.encode("utf-8"))
    temp = assignee+"*pawan*"+summary+"\n"
    report.write (temp)
def direct_load():    
    url = 'https://bugs.eclipse.org/bugs/buglist.cgi?bug_status=ASSIGNED&limit=0&order=bug_id&query_format=advanced'
    while True:
        try : 
            response = requests.get(url) 
            break
        except requests.exceptions.RequestException:
            time.sleep(300)
    tree = etree.fromstring(response.text, parser)
    for a in tree.xpath("//*[@id]/td[1]/a"):
        href = ''.join(a.xpath("./@href"))
        bug_url = "https://bugs.eclipse.org/bugs/"+href
        create_report (bug_url)  
if __name__ == "__main__":
    report = open(report_name,'w+')
    #direct_load()
    indirect_load(FileList)
    report.close()
