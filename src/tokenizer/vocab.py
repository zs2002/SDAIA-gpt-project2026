"""Bidirectional token-to-ID vocabulary mapping."""


class Vocabulary:
    """Bidirectional mapping between string tokens and integer IDs.

    Unknown tokens resolve to the ``<unk>`` special token (ID 3 by default).
    """

    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def add_token(self, token: str) -> int:
        """Add *token* and return its integer ID.

        If the token is already present the existing ID is returned (no-op).
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        return new_id

    def get_id(self, token: str) -> int:
        """Return the ID for *token*, or the ``<unk>`` ID if not found."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        return self.token_to_id.get(self.UNK_TOKEN, -1)

    def get_token(self, token_id: int) -> str:
        """Return the token for *token_id*, or ``<unk>`` if not found."""
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        return self.UNK_TOKEN

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.token_to_id)
