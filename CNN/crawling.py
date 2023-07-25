from selenium import webdriver
from selenium.webdriver.common.by import By
import os, urllib.request, datetime, chromedriver_autoinstaller, splitfolders

class GoogleImageCrawler:
    """
    keywords : 다운받을 이미지 키워드 리스트를 입력받습니다. \n
    num_of_data : 다운받을 이미지의 개수를 정합니다. \n
    scroll_seconds : 이미지 페이지에서 스크롤 다운 하는 시간을 설정합니다. 기본값은 40초입니다. \n
    path_datasets : 데이터셋을 저장할 경로입니다. 따로 지정하지 않을 경우 같은 경로에 datasets라는 이름으로 폴더를 생성합니다. \n
    path_output_datasets : 데이터셋을 train, test, validation으로 분할한 이후 결과를 해당 변수의 경로에 저장합니다. 따로 지정하지 않을 경우 \n
                            같은 경로에 outputs라는 폴더에 저장합니다. \n
    train, test, valid_size : 훈련, 테스트, 검증셋으로 분할 시 비율을 설정합니다. 기본 설정비율은 7:2:1 입니다.         
    """
    def __init__(self, keywords: list, 
                 num_of_data: int, 
                 scroll_seconds: int = 40, 
                 path_datasets: str = "./datasets", 
                 path_output_datasets: str = "./outputs",
                 train_size: float = 0.7,
                 test_size: float = 0.2,
                 valid_size: float = 0.1):
        
        self.keywords = keywords
        self.num_of_data = num_of_data
        self.scroll_seconds = scroll_seconds
        self.path_datasets = path_datasets
        self.path_output_datasets = path_output_datasets
        self.train_size = train_size
        self.test_size = test_size
        self.valid_size = valid_size
        
    def chrome_driver_download(self):
        """
        chromedriver_autoinstaller를 활용하여 chrome driver를 자동으로 다운받습니다. 이미 설치되어 있는 경우, 업데이트를 진행합니다.
        """
        chromedriver_autoinstaller.install(True)
        print("크롬 드라이버를 설치했습니다.")
        
    def set_options(self):
        """
        selenium의 옵션들을 설정합니다.
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--disable-gpu')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option('excludeSwitches', ['enable-automation'])

        return options
    
    def sroll_n_seconds(self, seconds):
        """
        설정하는 시간만큼, 해당 키워드로 검색된 이미지 페이지에서 스크롤 다운 합니다.
        """
        self.start = datetime.datetime.now()
        self.end = self.start + datetime.timedelta(seconds=seconds)
        
        while True:
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            try:
                self.driver.find_element(By.CLASS_NAME, "LZ4I").click()
            except:
                pass
            
            if datetime.datetime.now() > self.end:
                break
    
    def crawl(self):
        """
        키워드들을 순회하면서 이미지 주소를 추출합니다. 추출한 결과는 {키워드:이미지주소 리스트} 의 dict로 리턴합니다.
        """
        options = self.set_options()
        self.driver = webdriver.Chrome(options=options)
        src_dict = {}
        for keyword in self.keywords:
            if " " in keyword:
                keyword_splited = keyword.split(" ")
                url_part = ""
                for key in keyword_splited: 
                    url_part += key + "+"
                    
                url = f"https://www.google.com/search?q={url_part[:-1]}&tbm=isch&source=lnms&sa=X&ved=2ahUKEwig-qng0amAAxUBmVYBHRa3AVMQ0pQJegQICRAB&biw=1365&bih=961&dpr=1"
            else:
                url = f"https://www.google.com/search?q={keyword}&tbm=isch&source=lnms&sa=X&ved=2ahUKEwig-qng0amAAxUBmVYBHRa3AVMQ0pQJegQICRAB&biw=1365&bih=961&dpr=1"
                
            self.driver.get(url)
            self.sroll_n_seconds(self.scroll_seconds)
            src_list = []
            imgs = self.driver.find_elements(By.TAG_NAME, "img")
            for img in imgs:
                try:
                    src = img.get_attribute("src")
                    src_list.append(src)
                except:
                    print("there are no any src attributes")
                
                if len(src_list) == self.num_of_data:
                    break
            src_dict[keyword] = src_list
        
        self.driver.close()
        return src_dict
    
    def make_directory(self, some_dict):
        """
        폴더를 생성하지 않은 경우, 모든 데이터를 포함하는 디렉토리를 하나 생성하고, 하위에 keyword로 이뤄진 폴더들을 생성합니다.
        """
        if not os.path.isdir(self.path_datasets):
            os.mkdir(self.path_datasets)
        
        for key in some_dict.keys():
            if not os.path.isdir(f"{self.path_datasets}/{key}"):
                os.mkdir(f"{self.path_datasets}/{key}")
        
    def dowload_datasets(self):
        """
        크롤링하여 받은 결과 dictionary 내부의 각 키값들에 대한 이미지 url 리스트를 순회하면서 이미지를 다운받습니다. \n
        크롤링 중 이미지 주소를 찾지 못할경우, 클래스 객체 생성 시 입력한 데이터 개수보다 적게 들어가는 경우도 존재합니다.
        """
        src_dict = self.crawl()
        self.make_directory(src_dict)
        for key in src_dict.keys():
            for idx, img_url in enumerate(src_dict[key]):
                try:
                    urllib.request.urlretrieve(img_url, f"{self.path_datasets}/{key}/{idx}_{key}.jpg")
                except:
                    pass
            print(f"{key} 이미지 {len(os.listdir(f'{self.path_datasets}/{key}/'))}개 중 \
                {self.num_of_data - len(os.listdir(f'{self.path_datasets}/{key}/'))}개 자료가 소실되었습니다.")
                
    def seperate_files(self):
        """
        데이터를 train, test, validation 셋으로 분할합니다. 기본 설정 비율은 7:2:1입니다. \n
        이후 디렉터리의 이름을 라벨값과 함께 붙인 이름으로 변경하여 라벨링합니다.
        """
        splitfolders.ratio(self.path_datasets, self.path_output_datasets, ratio = (self.train_size, self.test_size, self.valid_size))
        for lst_dir in os.listdir(f"{self.path_output_datasets}/"):
            for idx, dir in enumerate(os.listdir(f"{self.path_output_datasets}/{lst_dir}/")):
                os.rename(f"{self.path_output_datasets}/{lst_dir}/{dir}", f"{self.path_output_datasets}/{lst_dir}/{idx}_{dir}")
            
if __name__ == "__main__":
    keywords = ["AirPlane Image", "Ship Images", "Car Images"] # 다운받을 이미지를 어떤 식으로 구글에 검색할지, 검색어를 리스트 안에 집어넣는다.
    my_crawler = GoogleImageCrawler(keywords, 100, 40, "./my_test", "./test_outputs") # 객체 생성
    my_crawler.chrome_driver_download() # 크롬드라이버 다운로드. 최초 실행 이후 주석처리해도 무관
    my_crawler.dowload_datasets() # 이미지 다운로드
    my_crawler.seperate_files() # 파일 분할