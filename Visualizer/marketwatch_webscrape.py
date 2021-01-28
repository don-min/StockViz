from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq

def scrape():
    '''
    returns a list of dictionaries containing the information of the top 3 articles at marketwatch.com

    [
        {link1: ..., title1: ..., headline1: ...},
        {link2: ..., title2: ..., headline2: ...},
        {link3: ..., title3: ..., headline3: ...}
    ]
    '''
    marketwatch = 'https://www.marketwatch.com/markets?mod=top_nav'

    uClient = uReq(marketwatch)
    page_html = uClient.read()
    uClient.close()

    page_soup = soup(page_html, "html.parser")

    articles = page_soup.findAll("div", {"class":"article__content"})

    # init the list
    article_info = []

    for article in articles[:3]:

        # init the dictionary
        keys = ['link', 'title', 'headline']
        article_dict = {key: None for key in keys}

        # link
        article_dict['link'] = article.h3.a['href']

        # title
        article_dict['title'] = article.h3.a.text.strip()

        # headline
        if article.p:
            headline = article.p.text
        elif not article.p:
            headline = 'No headline provided...'

        article_dict['headline'] = headline

        # append to the article_info list
        article_info.append(article_dict)

    return article_info

if __name__ == '__main__':
    print(scrape())

    # try commenting line 53 and run the codes below to see one by one
    #print("\n")
    #print(scrape()[0])
    #print("\n")
    #print(scrape()[1])
    #print("\n")
    #print(scrape()[2])
