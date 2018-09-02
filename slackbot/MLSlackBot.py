from slackbot.SlackBot import SlackBot, SlackBotFsm


class MLSlackBot(SlackBot):
    def __init__(self, token_path='./slackbot/tokens/ml_bot_token', channel='mltool_bot'):
        super().__init__(token_path, channel)
        self.fsm = SlackBotFsm()
        self._title = None
        self._msg_body = None

    @property
    def title(self):
        return self._title

    def send_start_msg(self, *args, **kwargs):
        pass

    def send_finish_msg(self, *args, **kwargs):
        pass

    def msg(self, head, body):
        return f"""
    {head}
    ------------------------------------
    {body}        
    """

    def msg_head(self, title, status, time, elapse_time):
        model_id = None
        epoch = None
        path = None
        params = None

        return f"""
    title  | {title}
    status | {status}
    time   | {time}
    elapsed| {elapse_time}
    model id| {model_id}
    train epoch | {epoch}
    model save path | {path}
    params | {params}
    """

    def error_msg_body(self, e):
        return f"""
    raise error {e}
    stack trace
    """

    def pending_msg_body(self, ):
        metric = None
        loss = None
        comment = None

        return f"""
    metric = {metric}
    loss = {loss}
    comment = {comment}
    """

    def finish_msg(self):
        param = None
        comment = None
        return f"""
    param = {param}
    comment = {comment}
    """
