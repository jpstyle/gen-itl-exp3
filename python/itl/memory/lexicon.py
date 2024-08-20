class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world.
    Allows many-to-many mappings between symbols and denotations.

    Types of considered denotations: perceivable unary concepts ("pcls"), perceivable
    binary concepts ("prel"), action concepts ("arel")
    """
    # Special reserved symbols; any predicates that are not straightforward to
    # represent as lexicon entry
    RESERVED = [
        # Invokes concept instance (i.e., set element) check
        ("sp", "isinstance"),
        # Invokes supertype-subtype check (against taxonomy KB)
        ("sp", "subtype"),
        # Represents agent lack of knowledge
        ("sp", "unknown"),
        # Represents agent inability
        ("sp", "unable"),
        # Represents 1st, 2nd person pronouns
        ("sp", "pronoun1"), ("sp", "pronoun2"),
        # Signals demonstration of how to achieve a task
        ("sp", "demonstrate"), ("sp", "manner")
    ]

    def __init__(self):
        self.s2d = {}     # Symbol-to-denotation
        self.d2s = {}     # Denotation-to-symbol

        # Add reserved symbols & denotations
        for r in Lexicon.RESERVED: self.add(r, r)

        # Specify the initial inventory of lexical entries that each agent starts with
        self.add(("vs", "have"), ("prel", 0))
        self.add(("va", "build"), ("arel", 0))
        self.add(("va", "pick_up"), ("arel", 1))
        self.add(("va", "pick_up_left"), ("arel", 2))
        self.add(("va", "pick_up_right"), ("arel", 3))
        self.add(("va", "drop_left"), ("arel", 4))
        self.add(("va", "drop_right"), ("arel", 5))
        self.add(("va", "assemble_right_to_left"), ("arel", 6))
        self.add(("va", "assemble_left_to_right"), ("arel", 7))
        self.add(("va", "inspect_left"), ("arel", 8))
        self.add(("va", "inspect_right"), ("arel", 9))

    def __repr__(self):
        return f"Lexicon(len={len(self.s2d)})"

    def __contains__(self, symbol):
        return symbol in self.s2d

    def add(self, symbol, denotation):
        # Symbol-to-denotation
        if symbol in self.s2d:
            self.s2d[symbol].append(denotation)
        else:
            self.s2d[symbol] = [denotation]
        
        # Denotation-to-symbol
        if denotation in self.d2s:
            self.d2s[denotation].append(symbol)
        else:
            self.d2s[denotation] = [symbol]
