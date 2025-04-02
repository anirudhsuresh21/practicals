import csv, requests, xml.etree.ElementTree as ET

def loadRSS():
    with open('topnewsfeed.xml', 'wb') as f:
        f.write(requests.get('http://feeds.bbci.co.uk/news/rss.xml').content)

def parseXML(xmlfile):
    root = ET.parse(xmlfile).getroot()
    return [{child.tag: child.text for child in item 
            if not child.tag.endswith(('thumbnail', 'content'))}
           for item in root.findall('.//item')]

def savetoCSV(newstems, filename):
    fields = ['guid', 'title', 'pubDate', 'description', 'link']
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(newstems)

if __name__ == "__main__":
    loadRSS()
    savetoCSV(parseXML('topnewsfeed.xml'), 'topnews.csv')
