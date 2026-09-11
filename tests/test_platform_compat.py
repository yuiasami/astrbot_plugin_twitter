"""平台兼容回归测试：合并转发按平台降级、UMO 平台反查与 WebUI 提示。"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from test_media_lifecycle import (
    Image,
    Nodes,
    Plain,
    _delivery_contract,
    _delivery_settings,
)
from test_media_lifecycle import (
    plugin_module as plugin_module,
)

ROOT = Path(__file__).resolve().parents[1]


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeMeta:
    def __init__(self, platform_id, name):
        self.id = platform_id
        self.name = name


class FakePlatform:
    def __init__(self, platform_id, name):
        self._meta = FakeMeta(platform_id, name)

    def meta(self):
        return self._meta


class FakeManager:
    def __init__(self, platforms):
        self._platforms = list(platforms)

    def get_insts(self):
        return list(self._platforms)


class DeliveryContext:
    """记录 send_message 调用并暴露平台实例列表。"""

    def __init__(self, platforms):
        self.sent = []
        self.platform_manager = FakeManager(platforms)

    async def send_message(self, _umo, message_chain):
        self.sent.append(message_chain.chain)


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [
        ("qq_official", False),
        ("QQ_OFFICIAL", False),
        ("qq_official_webhook", False),
        ("aiocqhttp", True),
        ("satori", True),
        ("", True),
        (None, True),
    ],
)
def test_supports_node_denylist(plugin_module, platform_name, expected):
    assert (
        plugin_module.TweetDeliveryService.supports_node(platform_name)
        is expected
    )


@pytest.mark.parametrize(
    ("use_node", "platform_name", "expected"),
    [
        (False, "aiocqhttp", False),
        (True, "qq_official", False),
        (True, "aiocqhttp", True),
    ],
)
def test_node_enabled_for_considers_config_and_platform(
    plugin_module,
    use_node,
    platform_name,
    expected,
):
    delivery = plugin_module.TweetDeliveryService(
        DeliveryContext([]),
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=use_node),
    )
    assert delivery.node_enabled_for(platform_name=platform_name) is expected


def test_umo_reverse_lookup_with_distinct_id(plugin_module):
    # 平台实例 ID 与类型名不同（qqbot-1 vs qq_official），UMO 首段是实例 ID。
    delivery = plugin_module.TweetDeliveryService(
        DeliveryContext([FakePlatform("qqbot-1", "qq_official")]),
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=True),
    )
    assert (
        delivery.platform_name_for_umo("qqbot-1:GroupMessage:o_123")
        == "qq_official"
    )
    assert delivery.node_enabled_for("qqbot-1:GroupMessage:o_123") is False


def test_unresolved_platform_retries_after_reload(plugin_module):
    context = DeliveryContext([])
    delivery = plugin_module.TweetDeliveryService(
        context,
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=True),
    )
    umo = "ghost:GroupMessage:1"
    assert delivery.platform_name_for_umo(umo) == ""
    # 平台热重载后出现了对应实例，不应被之前的空结果卡住。
    context.platform_manager = FakeManager(
        [FakePlatform("ghost", "qq_official")]
    )
    assert delivery.platform_name_for_umo(umo) == "qq_official"


def test_resolved_platform_is_cached(plugin_module):
    context = DeliveryContext([FakePlatform("qqbot-1", "qq_official")])
    delivery = plugin_module.TweetDeliveryService(
        context,
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=True),
    )
    umo = "qqbot-1:GroupMessage:o_123"
    assert delivery.platform_name_for_umo(umo) == "qq_official"
    # 命中缓存后即使平台实例消失也不会重新反查。
    context.platform_manager = FakeManager([])
    assert delivery.platform_name_for_umo(umo) == "qq_official"


@pytest.mark.asyncio
async def test_prepare_event_delivery_degrades_for_qq_official(plugin_module):
    chain = [
        Plain("tweet text"),
        Image.fromURL("https://example.com/image.jpg"),
    ]
    delivery = plugin_module.TweetDeliveryService(
        DeliveryContext([]),
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=True),
    )

    plain = delivery.prepare_event_delivery(chain, "Tester", "qq_official")
    # 文字优先：主链只含文字，图片拆到独立消息链，按顺序先发文字再发图片。
    assert [type(component) for component in plain.primary_chain] == [Plain]
    assert [type(component) for component in plain.media_chains[0]] == [Image]

    node = delivery.prepare_event_delivery(chain, "Tester", "aiocqhttp")
    assert len(node.primary_chain) == 1
    assert isinstance(node.primary_chain[0], Nodes)
    assert node.media_chains == []


@pytest.mark.asyncio
async def test_send_to_subscriber_never_sends_nodes_for_qq_official(
    plugin_module,
):
    class Messages:
        async def build_message_chain(self, *_args, **_kwargs):
            return [
                Plain("tweet"),
                Image.fromURL("https://example.com/image.jpg"),
            ]

    context = DeliveryContext([FakePlatform("qqbot-1", "qq_official")])
    delivery = plugin_module.TweetDeliveryService(
        context,
        object(),
        Messages(),
        _delivery_settings(plugin_module, use_node=True),
    )
    sent = await delivery.send_to_subscriber(
        "qqbot-1:GroupMessage:o_123",
        "tester",
        {},
        {"status": True, "r18": True, "media": False},
        "@tester (Tester)",
    )

    assert sent is True
    # 文字优先：先发文字消息，再发图片消息，且不出现合并转发组件。
    assert [type(component) for chain in context.sent for component in chain] == [
        Plain,
        Image,
    ]
    assert len(context.sent) == 2
    assert context.sent[0][0].text == "tweet"
    assert isinstance(context.sent[1][0], Image)
    assert not any(
        isinstance(component, Nodes)
        for chain in context.sent
        for component in chain
    )


@pytest.mark.asyncio
async def test_send_plain_chain_separate_media_text_first(plugin_module):
    context = DeliveryContext([FakePlatform("qqbot-1", "qq_official")])
    delivery = plugin_module.TweetDeliveryService(
        context,
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=True),
    )
    chain = [
        Plain("tweet text"),
        Image.fromURL("https://example.com/a.jpg"),
        Image.fromURL("https://example.com/b.jpg"),
    ]

    sent = await delivery.send_plain_chain_resilient(
        "qqbot-1:GroupMessage:o_123",
        chain,
        separate_media=True,
    )

    assert sent is True
    assert [
        type(component) for chain_ in context.sent for component in chain_
    ] == [Plain, Image, Image]
    assert [len(chain_) for chain_ in context.sent] == [1, 1, 1]
    assert isinstance(context.sent[0][0], Plain)


@pytest.mark.asyncio
async def test_send_plain_chain_keeps_single_message_by_default(plugin_module):
    context = DeliveryContext([FakePlatform("qqbot-1", "qq_official")])
    delivery = plugin_module.TweetDeliveryService(
        context,
        object(),
        object(),
        _delivery_settings(plugin_module, use_node=True),
    )
    chain = [Plain("tweet text"), Image.fromURL("https://example.com/a.jpg")]

    sent = await delivery.send_plain_chain_resilient(
        "qqbot-1:GroupMessage:o_123",
        chain,
    )

    assert sent is True
    assert len(context.sent) == 1
    assert [type(component) for component in context.sent[0]] == [
        Plain,
        Image,
    ]


@pytest.mark.asyncio
async def test_collective_forward_sends_immediately_for_qq_official(
    plugin_module,
):
    subscriptions_data = {
        "tester": {
            "screen_name": "Tester",
            "subscribers": {
                "qqbot-1:GroupMessage:o_123": {
                    "status": True,
                    "r18": True,
                    "media": False,
                }
            },
        }
    }

    class Subscriptions:
        async def get_all(self):
            return subscriptions_data

        async def get_retweet_seen(self):
            return {}

        async def save_retweet_seen(self, _data):
            return None

    class Messages:
        @staticmethod
        def build_nickname(username, screen_name):
            return f"@{username} ({screen_name})"

        @staticmethod
        def build_author_display(username, screen_name):
            return f"@{username} ({screen_name})"

        @staticmethod
        def tweet_has_media(_tweet_info):
            return False

        async def maybe_translate(self, _tweet_info, _umo, cycle=None):
            return None, None

        async def build_message_chain(self, *_args, **_kwargs):
            return [Plain("tweet")]

    context = DeliveryContext([FakePlatform("qqbot-1", "qq_official")])
    delivery = plugin_module.TweetDeliveryService(
        context,
        Subscriptions(),
        Messages(),
        _delivery_settings(
            plugin_module,
            use_node=True,
            collective_forward=True,
        ),
    )
    result = await delivery.push_to_subscribers(
        "tester",
        {
            "tweet_id": "1",
            "username": "tester",
            "screen_name": "Tester",
            "text": "tweet",
        },
    )

    contract = _delivery_contract(plugin_module)
    assert result.state is contract.DeliveryState.DELIVERED
    assert result.counts_toward_limit
    assert delivery.has_collected is False
    assert context.sent


def _load_webui_module():
    module_name = "twitter_webui_platform_test"
    sys.modules.pop(module_name, None)

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    web = types.ModuleType("astrbot.api.web")

    api.logger = _Logger()
    web.request = types.SimpleNamespace(json=lambda default=None: default)
    web.json_response = lambda data: (data, 200)
    web.error_response = lambda message, status_code=400: (
        {"message": message},
        status_code,
    )
    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.web": web,
        }
    )

    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "twitter_webui.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def webui_module():
    keys = ("astrbot", "astrbot.api", "astrbot.api.web")
    previous = {key: sys.modules.get(key) for key in keys}
    try:
        yield _load_webui_module()
    finally:
        # 恢复被替换的 astrbot 桩模块，避免泄漏影响后续测试文件。
        for key in keys:
            if previous[key] is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous[key]


class FakeWebContext:
    def __init__(self, platforms):
        self.routes = []
        self.platform_manager = types.SimpleNamespace(
            platform_insts=list(platforms)
        )

    def register_web_api(self, route, handler, methods, description):
        self.routes.append((route, handler, methods, description))


class FakeWebPlugin:
    async def _get_subscriptions_snapshot(self):
        return {}


@pytest.mark.asyncio
async def test_webui_rejects_qq_official_add_with_hint(webui_module):
    controller = webui_module.TwitterWebUIController(
        FakeWebPlugin(),
        FakeWebContext([FakePlatform("qqbot-1", "qq_official")]),
    )
    assert (
        await controller._validate_live_group(
            ("qqbot-1", "GroupMessage", "o_123")
        )
        == ("这个平台无法获取群列表，请在群内使用 /推特关注 指令新增订阅", 400)
    )


@pytest.mark.asyncio
async def test_webui_missing_platform_instance(webui_module):
    controller = webui_module.TwitterWebUIController(
        FakeWebPlugin(),
        FakeWebContext([]),
    )
    assert (
        await controller._validate_live_group(
            ("qqbot-1", "GroupMessage", "o_123")
        )
        == ("未找到对应的平台实例", 400)
    )
