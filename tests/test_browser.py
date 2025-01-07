import unittest
from tongagent.tools.browser import SimpleTextBrowser
from tongagent.tools.web_surfer import WebQATool
from TongAgent.tongagent.llm_engine.gpt import get_tonggpt_open_ai_client
import os
import tiktoken
from tqdm import tqdm
import json

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class TestBrowser(unittest.TestCase):
    def test_case(self):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        os.environ["SERPAPI_API_KEY"]= '5d595dbebc0c1f6b2c637ae1650402baf1e1f121f6cc8fce58928c1875e4fff4'
        browser_config = {
            "viewport_size": 1024 * 5,
            "downloads_folder": "coding",
            "request_kwargs": {
                "headers": {"User-Agent": user_agent},
                "timeout": 300,
            },
        }
        browser = SimpleTextBrowser(**browser_config)
        browser.visit_page(
            "https://en.wikipedia.org/wiki/Union_City,_Missouri"
        )
        
        print(num_tokens_from_string(browser.page_content, "cl100k_base"))
        print(num_tokens_from_string(browser.viewport, "cl100k_base"))
        print(len(browser.page_content), len(browser.viewport))
    
    def test_case2(self):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        os.environ["SERPAPI_API_KEY"]= '5d595dbebc0c1f6b2c637ae1650402baf1e1f121f6cc8fce58928c1875e4fff4'
        browser_config = {
            "viewport_size": 1024 * 5,
            "downloads_folder": "coding",
            "request_kwargs": {
                "headers": {"User-Agent": user_agent},
                "timeout": 300,
            },
        }
        from datasets import load_from_disk
        data_path = "/home/bofei-zhang/Documents/Project/Projects/Multimodal-CL/iclr_09/wiki/data/subset_1000"
        dataset = load_from_disk(data_path)
        browser = SimpleTextBrowser(**browser_config)
        output = []
        # dataset = dataset[:100]
        for item in tqdm(dataset):
            # print(item)
            url = item["url"]
            item_id = item["id"]
            browser.visit_page(
                url
            )
            total_token = num_tokens_from_string(browser.page_content, "cl100k_base")
            output.append(
                [url, total_token]
            )
            if len(output) == 100:
                break
            #print(num_tokens_from_string(browser.viewport, "cl100k_base"))
            #print(len(browser.page_content), len(browser.viewport))
        with open("tmp.json", "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
    
    def test_case3(self):
        from tongagent.tools.web_surfer import browser
        browser.visit_page("https://en.wikipedia.org/wiki/Mercedes_Sosa")
        tool = WebQATool()
        question = "hat are the studio albums released by Mercedes Sosa between 2000 and 2009?"
        result = tool.forward(question)
        print("Output", result)
        
    def test_case4(self):
        from tongagent.tools.web_surfer import browser
        from tongagent.tools.browser import google_custom_search
        from tongagent.utils import load_config
        config = load_config()
        cx, key = config.search_engine[0].cx, config.search_engine[0].key
        print(google_custom_search("Bielefeld Academic Search Engine BASE", cx, key))
    
    
if __name__ == "__main__":
    unittest.main()