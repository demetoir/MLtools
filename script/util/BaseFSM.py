class BaseFSM:
    # TODO add logger mixIN?
    # TODO auto complete able event function

    def __init__(self):
        self.initial_state = None
        self._state = None
        self._states = []
        self._events = []
        self._state_machine = {}
        self._state_to_str = {}
        self._state_to_num = {}
        self._event_to_str = {}
        self._event_to_num = {}
        self._check_state = {}
        self._check_event = {}

    @property
    def state(self):
        return self._state

    @property
    def get_events(self):
        return self._events

    @property
    def states(self):
        return self._states

    @property
    def get_state_machine(self):
        return self._state_machine

    @property
    def state_to_str(self):
        return self._state_to_str

    @property
    def state_to_num(self):
        return self._state_to_num

    @property
    def event_to_num(self):
        return self._event_to_num

    @property
    def event_to_str(self):
        return self._event_to_str

    def _update_state(self, new_state):
        if type(new_state) == int:
            self._state = self.state_to_str[new_state]
        elif type(new_state) == str:
            self._state = self._check_state[new_state]
        else:
            raise ValueError(f'{new_state} does not expected state')

    def _raise_event(self, event):
        if type(event) is int:
            event = self._event_to_str[event]
        elif type(event) is str:
            event = self._check_event[event]
        else:
            raise ValueError(f'{event} does not expected event')

        self._state = self._state_machine[self._state][event]

    def add_state(self, name, num=None, initial_state=False):
        if num is None:
            num = len(self.states)

        if initial_state:
            self.initial_state = name
            self._state = name

        self._states += [name]
        self._state_to_num[name] = num
        self._state_to_str[num] = name
        self._state_machine[name] = {}
        self._check_state[name] = name

    def add_event(self, name, source, destination, num=None):
        if num is None:
            num = len(self._events)

        if source not in self.states:
            raise ValueError(f"""'{source}' state is not in states""")
        if destination not in self.states:
            raise ValueError(f"""'{destination}' state is not in state""")

        self._events += [name]
        self._event_to_str[num] = name
        self._event_to_num[name] = num
        self._state_machine[source][name] = destination
        self._check_event[name] = name

        def event_func():
            self._raise_event(name)

        event_func.__name__ = name
        setattr(self, name, event_func)
