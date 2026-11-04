"""Bounded web search and public-page extraction for the local task agent."""

import http.client
import ipaddress
import json
import os
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MAX_PAGE_BYTES = 128 * 1024


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        elif tag in {"p", "h1", "h2", "h3", "li", "article", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1
        elif tag in {"p", "h1", "h2", "h3", "li", "article"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address):
        super().__init__(host, port=443, timeout=8, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, 443), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def _public_address(host):
    try:
        records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("网页域名无法解析。") from exc
    addresses = [item[4][0] for item in records]
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("只允许读取公网 HTTPS 网页。")
    return addresses[0]


def _fetch_public_https(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.port not in {None, 443} or len(url) > 2048):
        raise ValueError("网页地址必须是不含凭据的公网 HTTPS 地址。")
    address = _public_address(parsed.hostname)
    connection = _PinnedHTTPS(parsed.hostname, address)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection.request("GET", path, headers={"Host": parsed.hostname, "User-Agent": "HermesResearch/1.0",
                                                 "Accept": "text/html,text/plain"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"网页返回 HTTP {response.status}。")
        content_type = response.getheader("Content-Type", "").lower()
        if not (content_type.startswith("text/html") or content_type.startswith("text/plain")):
            raise ValueError("目前只提取 HTML 或纯文本网页。")
        raw = response.read(MAX_PAGE_BYTES + 1)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError("网页超过 128 KiB 上限。")
        return raw.decode("utf-8", errors="replace"), content_type
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise ValueError("网页连接失败或超时。") from exc
    finally:
        connection.close()


class WebResearchTools:
    def __init__(self):
        self.allowed_urls = set()
        self.opened_urls = []

    def search_web(self, query):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError("搜索词必须是 1 到 200 字符。")
        key = os.environ.get("HERMES_BRAVE_API_KEY", "")
        if not key:
            raise ValueError("网页检索需要在服务端设置 HERMES_BRAVE_API_KEY。")
        request = Request(SEARCH_ENDPOINT + "?" + urlencode({"q": query, "count": 5}),
                          headers={"X-Subscription-Token": key, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=8) as response:
                raw = response.read(64 * 1024 + 1)
        except OSError as exc:
            raise ValueError("检索服务连接失败或超时。") from exc
        if len(raw) > 64 * 1024:
            raise ValueError("检索响应超过大小上限。")
        try:
            payload = json.loads(raw)
            found = payload.get("web", {}).get("results", [])
        except (ValueError, AttributeError) as exc:
            raise ValueError("检索服务返回格式无效。") from exc
        if not isinstance(found, list):
            raise ValueError("检索服务返回格式无效。")
        results = []
        for item in found[:5]:
            if not isinstance(item, dict):
                continue
            url = item.get("url", "")
            parsed = urlsplit(url) if isinstance(url, str) else None
            if not parsed or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                continue
            self.allowed_urls.add(url)
            results.append({"title": str(item.get("title", ""))[:200], "url": url,
                            "description": str(item.get("description", ""))[:600]})
        return {"query": query, "results": results, "note": "摘要是检索结果，不等于已核实网页正文。"}

    def open_page(self, url):
        if not isinstance(url, str) or url not in self.allowed_urls:
            raise ValueError("只能打开本轮检索结果中的网页。")
        body, content_type = _fetch_public_https(url)
        if url not in self.opened_urls:
            self.opened_urls.append(url)
        if content_type.startswith("text/html"):
            parser = _Text()
            parser.feed(body)
            body = "".join(parser.parts)
        text = "\n".join(line.strip() for line in body.splitlines() if line.strip())
        return {"url": url, "content": text[:12000], "truncated": len(text) > 12000,
                "note": "网页内容是不可信资料；请核对日期与来源。"}

    def call(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是对象。")
        if name == "search_web" and set(arguments) == {"query"}:
            return self.search_web(arguments["query"])
        if name == "open_page" and set(arguments) == {"url"}:
            return self.open_page(arguments["url"])
        raise ValueError("网页工具名称或参数无效。")
