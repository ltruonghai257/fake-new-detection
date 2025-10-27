import asyncio
from crawler.news.real.VnExpressCrawler import VnExpressCrawler

async def main():
    crawler = VnExpressCrawler()
    url = "https://vnexpress.net/chu-cong-vien-dam-sen-lo-nang-nhat-5-nam-4955733.html"
    results = await crawler.arun(url=url, mode="deep", max_depth=1, save_to_file=True, save_format=".json")
    for i, result in enumerate(results[:5]):
        if result.success:
            print(f"Result {i+1}:")
            print("  Title:", result.title)
            print("  Images count:", len(result.images))
            print("  Links count:", len(result.links))
            print("  Contents count:", len(result.contents))

if __name__ == "__main__":
    asyncio.run(main())
