from bs4 import BeautifulSoup as Soup
import requests
import pandas as pd
import time
import datetime
import os

today = datetime.date.today()
y, m, d = today.year, today.month, today.day
page = 1 
column_names = ['X', 'Filling', 'Trade', 'tick', 'company_name', 'insider_name',
                'title', 'type', 'price', 'qty', 'owned', 'delta_owned', 'value']
output_path = 'oi_csv.csv'


if os.path.exists(output_path):
    os.remove(output_path)

while int(y) > 2000:
    start = time.time()
    max_date = f'{m}%2F{d}%2F{y}'
    print(f'Start {d}.{m}.{y} Start time: {time.strftime("%H:%M:%S", time.gmtime(start))}', end='\r\r')
    url1 = f'http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=-1&fdr=01%2F01%2F1999+-+{max_date}&td=0&td' \
           'r=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&xa=1&xd=1&xg=1&xf=1&xm=1&xx=1&xc=1&xw=1&excludeDerivRelated=1&vl=&vh' \
           '=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol' \
           f'=0&cnt=1000&page={page}'

    get_result = requests.get(url1, timeout=1000).text
    soup = Soup(get_result, 'html.parser')

    table = soup.find("table", {"class": "tinytable"})

    result_table = []

    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        try:
            new_row = [tds[i].text for i in range(len(tds))][0:-4]
            result_table.append(new_row)
        except IndexError as err:
            pass

    small_oi_frame = pd.DataFrame(result_table, columns=column_names)
    new_date = small_oi_frame.tail(1)['Filling'].to_list()[0][0:10].split('-')

    if new_date == [y, m, d]:
        page += 1
    else:
        y, m, d = new_date
        page = 1

    small_oi_frame.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))
    elapsed = time.time() - start

    print(f'Done {d}.{m}.{y} Elapsed time: {elapsed} Rows:{len(small_oi_frame)}', end='\r')

    time.sleep(abs(10 - elapsed)%10)


