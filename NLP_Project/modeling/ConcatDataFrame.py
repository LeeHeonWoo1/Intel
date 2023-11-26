"""
새로 크롤링한 데이터프레임을 기존에 사용하던 전처리 기법을 활용하여 병합합니다.
"""
import pandas as pd
from typing import Union
import time
import numpy as np
from selenium import webdriver
import csv
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.by import By
from sklearn.preprocessing import StandardScaler


class ConcatDataFrame:
    """
    전처리, 크롤링, 데이터 프레임 병합을 진행합니다. 받는 인자값은 아래와 같습니다.

    df : title, views 두 개의 컬럼명을 포함하는 기존에 존재하는 데이터프레임입니다. \n
    crawl_cafe_url : 네이버 카페 주소입니다. \n
    crawl_club_id : 네이버 카페 게시판의 id값입니다. \n
    save_csv_path : 크롤링을 진행한 후 저장될 경로입니다. \n
    new_df : 새로 병합할 데이터프레임입니다.
    """

    def __init__(self, df: Union[pd.DataFrame, None] = None, new_df: Union[pd.DataFrame, None] = None, crawl_cafe_url: Union[None, str, list] = None,
        crawl_club_id: Union[None, int, list] = None, save_csv_path: Union[None, str] = None, article_id: Union[None, int, list] = None):
        self.df = df
        self.new_df = new_df
        self.crawl_cafe_url = crawl_cafe_url
        self.crawl_club_id = crawl_club_id
        self.save_csv_path = save_csv_path
        self.article_id = article_id

    def pretreate(self, df: Union[pd.DataFrame, None]):
        """
        전처리를 진행합니다.
        """
        print("================================================")
        print("전처리를 시작합니다.")

        if self.crawl_cafe_url == None:
            work_df = df[["title", "views"]]
        else:
            work_df = pd.read_csv(self.save_csv_path)

        if work_df.isna().sum().sum() > 0:
            work_df.dropna(inplace=True, axis=0)

        if work_df["views"].dtype == "float64":
            work_df["views"] = work_df["views"].astype("string")
            views_int_count = (
                work_df["views"]
                .loc[work_df["views"].str.endswith(".0") == False]
                .count()
            )

            if views_int_count == 0:
                work_df["views"] = work_df["views"].astype(float).astype(int)

            else:
                work_df["views"] = work_df["views"].apply(lambda x: np.round(x))

        elif work_df["views"].dtype == "O":
            work_df['views'] = work_df['views'].apply(lambda x : x.replace(",", "") if "," in x else str(int(float(x.replace("만", "")))*1000))
            
            views_float_count = (
                work_df["views"]
                .loc[work_df["views"].str.endswith(".0") == True]
                .count()
            )

            if views_float_count > 0:
                work_df["views"] = (
                    work_df["views"].astype("float64").apply(lambda x: np.round(x))
                )

            else:
                work_df["views"] = work_df["views"].astype(int)
                
        origin_title_list = work_df['title'].values
        new_title_list = []
        for title in origin_title_list:
            if '[' in title and ']' in title:
                new_title = title.replace(title[ title.find('[') : title.find(']') + 1 ], "")
                new_title_list.append(new_title)
            else:
                new_title_list.append(title)

        work_df['title'] = new_title_list

        try:
            work_df["title"] = work_df["title"].str.replace("[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣|a-zA-Z ]", "", regex=True)

        except:
            work_df["title"] = work_df["title"].replace("[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣|a-zA-Z ]", "", regex=True)

        dup_cnt = work_df["title"].loc[work_df["title"].duplicated() == True].count()

        if dup_cnt > 0:
            work_df.drop_duplicates(subset=["title"], inplace=True)
            work_df = work_df.reset_index(drop=True)

        work_df.dropna(axis = 0, inplace=True)

        print("전처리 완료")
        work_df.to_csv("./new_work_df.csv", sep = ",", index = False, encoding = "utf8")

        return work_df

    def crawl(self):
        """
        입력한 주소와 클럽 id, csv path를 기반으로 크롤링을 진행합니다.
        """

        print("================================================")
        print("입력한 주소로 크롤링을 시작합니다.")
        browser = webdriver.Chrome()

        idx = 0

        all_data = []
        for _ in range(0, 1000):
            row_data = []
            baseurl = self.crawl_cafe_url
            clubid = self.crawl_club_id
            articleid = self.article_id
            # for i in range(len(baseurl)):
            browser.get(baseurl)

            idx += 1

            boardtype = "L"
            pageNum = idx
            userDisplay = 50

            time.sleep(3)

            browser.get(baseurl + "ArticleList.nhn?search.clubid=" + str(clubid) + "&search.boardtype=" + str(boardtype)
                + "&search.page=" + str(pageNum) + "&userDisplay=" + str(userDisplay) + "&articleid=" + str(articleid)
            )
            
            articleid -= 1

            soup = bs(browser.page_source, "html.parser")
            data = soup.find_all(class_="ArticleContentBox")
            
            article_title = data.find(class_="title_text")
            article_content = data.find(class_="se-module.se-module-text")
            article_view = data.select_one("#app > div > div > div.ArticleContentBox > div.article_header > div.WriterInfo > div.profile_area > div.article_info > span.count")

            article_contents = article_content.find_all("p")
            main_context = ''
            for content in article_contents:
                main_context += content.text
            
            row_data.append(main_context)
            row_data.append(article_title.text)
            row_data.append(article_view.text)
            all_data.append(row_data)

        df = pd.DataFrame(all_data, columns=['content', 'title', 'views'])
        df.to_csv("./get_title_content.csv", sep = ",", index=False, encoding="utf8")
        print("크롤링 완료")

    def labeling(self, df):
        """
        라벨값 간 분포가 균등한 값을 찾아 라벨링을 진행합니다. 이 때 사용하는 기준값은 평균, 제1사분위수, 제2사분위수입니다.
        """

        # work_df = self.pretreate(df)
        work_df = df
        mean_val = work_df['views'].mean()
        
        work_df["label"] = self.get_label_list(work_df["views"].values, mean_val)

        if (abs(work_df["label"].value_counts()[0] - work_df["label"].value_counts()[1]) >= 250):
            quantile_values = [
                work_df["views"].quantile(0.25),
                work_df["views"].quantile(0.5),
            ]

            for value in quantile_values:
                work_df["label"] = self.get_label_list(work_df["views"].values, value)
                if (abs(work_df["label"].value_counts()[0] - work_df["label"].value_counts()[1]) < 250):
                    break
                
        work_df.to_csv("./test.csv", encoding="utf8", sep=",", index=False)
        # work_df = self.pretreate(df)
        # work_df = df
        # work_df['views'] = work_df['views'].astype('string')
        # df_value = work_df['views'].apply(lambda x : x.replace(",", "").replace("만", "")).values
        
        # df_value = df_value.reshape(-1, 1)
        # scaler = StandardScaler()
        # scaler.fit(df_value)
        
        # work_df['views'] = list(scaler.transform(df_value).reshape(-1))
        
        # label_list = []
        # for value in work_df['views'].values:
        #     if value >= 0.75:
        #         label_list.append(0)
        #     elif value >= 0.5:
        #         label_list.append(1)
        #     elif value >= 0.25:
        #         label_list.append(2)
        #     else:
        #         label_list.append(3)
        
        # work_df['label'] = label_list
        
        # work_df.to_csv("./test.csv", encoding="utf8", sep=",", index=False)
        
        return work_df

    def get_label_list(self, values, standard):
        """
        라벨 컬럼을 위한 리스트를 생성합니다.
        """

        label_list = []
        for value in values:
            if value >= standard:
                label_list.append(1)
            else:
                label_list.append(0)

        return label_list

    def concat(self):
        new_work_df = self.labeling(self.new_df)
        new_df = pd.concat([self.df, new_work_df], axis = 0)
        
        mean_val = int(new_df["views"].mean())
        new_df["label"] = self.get_label_list(new_df["views"].values, mean_val)

        if (
            abs(new_df["label"].value_counts()[0] - new_df["label"].value_counts()[1])
            >= 250
        ):
            quantile_values = [
                new_df["views"].quantile(0.25),
                new_df["views"].quantile(0.5),
            ]

            for value in quantile_values:
                new_df["label"] = self.get_label_list(new_df["views"].values, value)
                if (
                    abs(
                        new_df["label"].value_counts()[0]
                        - new_df["label"].value_counts()[1]
                    )
                    < 250
                ):
                    break
        new_df.drop_duplicates(subset=["title"], inplace=True, keep='first')
        new_df.to_csv("./new_df.csv", sep=",", encoding="utf8", index=False)
        
        print(f"{new_df.shape[0]}건의 데이터를 병합하였습니다.")
        print(f"새로운 데이터프레임의 라벨값 간 분포 : {new_df['label'].value_counts()[0]}, {new_df['label'].value_counts()[1]}")
        