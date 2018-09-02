from slacker import Slacker
from script.util.BaseFSM import BaseFSM


class SlackBotFsm(BaseFSM):
    def __init__(self):
        super().__init__()

        self.add_state('pending', initial_state=True)
        self.add_state('on going')
        self.add_state('finish')
        self.add_state('error')

        self.add_event('raise_error', 'pending', 'error')
        self.add_event('start', 'pending', 'on going')

        self.add_event('raise_error', 'on going', 'error')
        self.add_event('finish', 'on going', 'finish')

        self.add_event('raise_error', 'finish', 'error')

    def start(self):
        self.start()

    def raise_error(self):
        self.raise_error()

    def finish(self):
        self.finish()


def test_slack_bot_fsm():
    fsm = SlackBotFsm()
    print(fsm.state)
    fsm.start()
    print(fsm.state)

    fsm.raise_error()
    print(fsm.state)

    fsm.finish()
    print(fsm.state)


class SlackBot:
    def __init__(self, token_path=None, channel=None):
        self.token_path = token_path
        self.channel = channel
        self.slacker = Slacker(self._get_token(self.token_path))

    def _get_token(self, token_path):
        with open(token_path, 'r') as f:
            token = f.readlines()
        return token

    def post_message(self, msg, attachments=None):
        # TODO to make usable
        if attachments:
            attachments_dict = dict()
            attachments_dict['pretext'] = "pretext attachments 블록 전에 나타나는 text"
            attachments_dict['title'] = "title 다른 텍스트 보다 크고 볼드되어서 보이는 title"
            attachments_dict['title_link'] = "https://corikachu.github.io"
            attachments_dict['fallback'] = "클라이언트에서 노티피케이션에 보이는 텍스트 입니다. attachment 블록에는 나타나지 않습니다"
            attachments_dict['text'] = "본문 텍스트! 5줄이 넘어가면 *show more*로 보이게 됩니다."
            attachments_dict['mrkdwn_in'] = ["text", "pretext"]  # 마크다운을 적용시킬 인자들을 선택합니다.
            attachments = [attachments_dict]

            self.slacker.chat.post_message(channel=self.channel, text='tetsetseetsetset', attachments=attachments)
        else:
            self.slacker.chat.post_message(self.channel, msg)


def test_SlackBot():
    bot = SlackBot()
    bot.post_message('hello world')
