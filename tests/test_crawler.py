from bs4 import BeautifulSoup

from face_search.crawler import extract_image_urls, extract_links, largest_from_srcset, normalize_url

SITE = "https://site.com/"


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


class TestLargestFromSrcset:
    def test_width_descriptors(self):
        assert largest_from_srcset("a.jpg 400w, b.jpg 800w, c.jpg 600w") == "b.jpg"

    def test_density_descriptors(self):
        assert largest_from_srcset("a.jpg 1x, b.jpg 2x") == "b.jpg"

    def test_single_url_without_descriptor(self):
        assert largest_from_srcset("only.jpg") == "only.jpg"

    def test_empty(self):
        assert largest_from_srcset("") is None

    def test_commas_inside_urls(self):
        # Real CNN srcset: splitting on "," would cut the URLs in half
        srcset = ("https://m.cnn.com/x.jpg?q=h_144,w_256,c_fill/f_webp 256w, "
                  "https://m.cnn.com/x.jpg?q=h_438,w_780,c_fill/f_webp 780w")
        assert largest_from_srcset(srcset) == "https://m.cnn.com/x.jpg?q=h_438,w_780,c_fill/f_webp"


def test_normalize_url_makes_absolute_and_drops_fragment():
    assert normalize_url("/img/a.jpg#top", "https://site.com/page") == "https://site.com/img/a.jpg"


class TestExtractImageUrls:
    def test_all_sources(self):
        html = """
            <meta property="og:image" content="https://site.com/og.jpg">
            <img src="/plain.jpg">
            <img src="placeholder.gif" data-src="/lazy.jpg">
            <img srcset="/small.jpg 200w, /big.jpg 800w">
            <picture><source srcset="/pic.webp 1x"></picture>
        """
        assert extract_image_urls(soup(html), SITE) == {
            "https://site.com/og.jpg", "https://site.com/plain.jpg",
            "https://site.com/placeholder.gif", "https://site.com/lazy.jpg",
            "https://site.com/big.jpg", "https://site.com/pic.webp",
        }

    def test_skips_inline_data_uris(self):
        assert extract_image_urls(soup('<img src="data:image/png;base64,AAAA">'), SITE) == set()


def test_extract_links_keeps_only_same_site_html_pages():
    html = """
        <a href="/a">kept</a>
        <a href="https://www.site.com/b#section">kept: www. and #fragment ignored</a>
        <a href="/doc.pdf">pdf</a> <a href="/R.PDF">upper case</a>
        <a href="/f.pdf?download=1">query string</a> <a href="/v.mp4">video</a>
        <a href="/article.html?source=newsletter.pdf">kept: .pdf only in the query</a>
        <a href="https://other.com/x">other site</a> <a href="mailto:a@b.c">mail</a>
    """
    assert extract_links(soup(html), SITE, SITE) == {
        "https://site.com/a",
        "https://www.site.com/b",
        "https://site.com/article.html?source=newsletter.pdf",
    }
