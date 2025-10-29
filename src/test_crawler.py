import asyncio

from crawler.crawler_factory import crawl_and_save_all


async def main():
    urls_to_test = [
        "https://baochinhphu.vn/lan-toa-nhung-gia-tri-van-hoa-tot-dep-chan-thien-my-cua-dan-toc-102230318225051612.htm",
        "https://thanhnien.vn/giao-duc-anh-ngu-phat-trien-manh-me-bat-chap-khung-hoang-hau-covid-19-185230320124322595.htm",
        "https://tuoitre.vn/ly-hoang-nam-vao-ban-ket-giai-m25-lucknow-an-do-2023032414280582.htm",
        "https://baochinhphu.vn/hlv-philippe-troussier-chinh-thuc-nhan-nhiem-vu-voi-bong-da-viet-nam-102230227143914537.htm",
        "https://thanhnien.vn/le-hoi-thanh-nien-the-he-moi-thich-ung-ket-noi-va-chia-se-185230325203534866.htm",
        "https://thanhnien.vn/hoc-thong-qua-choi-kich-thich-su-sang-tao-o-hoc-sinh-tieu-hoc-185230322111015524.htm",
        "https://tuoitre.vn/hai-nguyen-pho-chu-tich-ha-noi-bi-de-nghi-xu-ly-trong-vu-nang-khong-gia-cay-xanh-20230328134424234.htm",
        "https://tuoitre.vn/evn-van-de-xuat-cap-dien-cho-con-dao-bang-cap-ngam-vuot-bien-20230317105713882.htm",
        "https://dantri.com.vn/giao-duc/ha-noi-khoang-50-hoc-sinh-tieu-hoc-nghi-ngo-doc-sau-khi-da-ngoai-20230328203127735.htm",
        "https://tuoitre.vn/giai-phap-thuc-day-so-hoa-truy-xuat-nguon-goc-nong-san-20230228134150083.htm",
        "https://tuoitre.vn/uy-ban-thuong-vu-quoc-hoi-khong-tan-thanh-quy-dinh-thoi-han-so-huu-chung-cu-20230317180028903.htm",
        "https://baochinhphu.vn/nam-2023-nhu-cau-hang-khong-phuc-hoi-ve-muc-truoc-dai-dich-covid-19-102230209144427102.htm",
        "https://dantri.com.vn/xa-hoi/mien-bac-mua-lanh-truoc-khi-don-dot-nang-nong-cuc-bo-20230330080713632.htm",
        "https://baochinhphu.vn/nong-nghiep-tiep-tuc-tang-truong-trong-quy-i-102230330180347503.htm",
        "https://dantri.com.vn/the-gioi/uav-ukraine-tap-kich-ban-chay-sieu-phao-nga-20230329003609965.htm",
        "https://baochinhphu.vn/nang-cap-thiet-bi-giam-sat-dao-tao-lai-xe-tu-dong-phat-hien-gian-lan-102230418163901398.htm",
        "https://tuoitre.vn/cac-tong-thong-my-dinh-vao-to-tung-20230331175157137.htm",
        "https://thanhnien.vn/anh-duoc-chap-nhan-gia-nhap-cptpp-185230331085115673.htm",
        "https://dantri.com.vn/the-gioi/giao-tranh-leo-thang-quanh-nha-may-ukraine-iaea-canh-bao-su-co-hat-nhan-20230330124018825.htm",
        "https://baochinhphu.vn/ha-noi-duoc-chon-la-diem-den-an-toan-cho-nu-du-khach-103230307135847784.htm",
        "https://vnexpress.net/chu-cong-vien-dam-sen-lo-nang-nhat-5-nam-4955733.html",
    ]

    await crawl_and_save_all(urls_to_test, "news_data.json")


if __name__ == "__main__":
    asyncio.run(main())
