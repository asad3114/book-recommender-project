"""A small, safe, personal desktop assistant that runs in your terminal.

Start it with ``python Jarvis.py``. Commands are matched locally; no API key is
needed. Notes are saved beside this file in ``jarvis_data/notes.json``.
"""

from __future__ import annotations

import ast
import json
import operator
import os
import base64
import re
import subprocess
import sys
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
NOTES_FILE = APP_DIR / "jarvis_data" / "notes.json"


def calculate(expression: str) -> int | float:
    """Evaluate basic arithmetic without eval() or access to Python objects."""
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("That exponent is too large.")
            result = operators[type(node.op)](left, right)
            if abs(result) > 10**100:
                raise ValueError("That result is too large.")
            return result
        raise ValueError("Use numbers and +, -, *, /, //, %, or ** only.")

    try:
        return visit(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("I couldn't calculate that expression.") from exc


class Jarvis:
    def __init__(self, voice: bool = False, ai: bool = False) -> None:
        self.running = True
        self.voice_enabled = voice
        self.ai_enabled = ai
        self.client = None
        self.previous_response_id: str | None = None
        self.commands: list[tuple[str, object]] = [
            (r"^(?:help|what can you do|commands)$", self.help),
            (r"^(?:hi|hello|hey)(?: jarvis)?[.! ]*$", self.greet),
            (r"^(?:time|what time is it|tell me the time)\??$", self.tell_time),
            (r"^(?:date|what(?:'s| is) the date|today(?:'s date)?)\??$", self.tell_date),
            (r"^(?:calculate|compute|what is)\s+(.+?)[?]?$", self.do_calculate),
            (r"^(?:search(?: the web)? for|google)\s+(.+)$", self.search_web),
            (r"^open\s+(google|youtube|github|wikipedia)$", self.open_site),
            (r"^(?:remember|note)\s+(.+)$", self.add_note),
            (r"^(?:show|read|list) my notes$", self.show_notes),
            (r"^(?:clear|delete) my notes$", self.clear_notes),
            (r"^ai on$", self.enable_ai),
            (r"^ai off$", self.disable_ai),
            (r"^reset conversation$", self.reset_conversation),
            (r"^(?:exit|quit|goodbye|stop)(?: jarvis)?[.! ]*$", self.quit),
        ]

    def say(self, message: str) -> None:
        print(f"Jarvis: {message}")
        if self.voice_enabled:
            self._speak(message)

    @staticmethod
    def _powershell(script: str, input_text: str = "", timeout: int = 15) -> str:
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            input=input_text, text=True, capture_output=True, timeout=timeout, check=False,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(detail or "Windows Speech could not start.")
        return result.stdout.strip()

    def _speak(self, message: str) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$text = [Console]::In.ReadToEnd(); "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Speak($text); $speaker.Dispose()"
        )
        try:
            self._powershell(script, message, timeout=30)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            print(f"(Voice output unavailable: {exc})")
            self.voice_enabled = False

    def _listen(self) -> str:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
            "$grammar = New-Object System.Speech.Recognition.DictationGrammar; "
            "$recognizer.LoadGrammar($grammar); $recognizer.SetInputToDefaultAudioDevice(); "
            "$result = $recognizer.Recognize([TimeSpan]::FromSeconds(8)); "
            "if ($result) { [Console]::WriteLine($result.Text) }; $recognizer.Dispose()"
        )
        try:
            return self._powershell(script, timeout=12)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            print(f"(Voice input unavailable: {exc})")
            self.voice_enabled = False
            return ""

    def _get_ai_client(self):
        if self.client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("Set the OPENAI_API_KEY environment variable first. See README.md.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the AI dependency with: python -m pip install -r requirements.txt") from exc
            self.client = OpenAI()
        return self.client

    def _ask_ai(self, text: str) -> None:
        try:
            client = self._get_ai_client()
            params = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-6-astra"),
                "instructions": "You are Jarvis, a helpful personal assistant. Be clear and concise.",
                "input": text,
            }
            if self.previous_response_id:
                params["previous_response_id"] = self.previous_response_id
            response = client.responses.create(**params)
            self.previous_response_id = response.id
            self.say(response.output_text or "I didn't get a text response that time.")
        except Exception as exc:
            self.say(f"AI request failed: {exc}")

    def enable_ai(self, _match: re.Match[str] | None = None) -> None:
        try:
            self._get_ai_client()
            self.ai_enabled = True
            self.say("AI chat is on. Ask me anything, or say 'ai off' to return to commands.")
        except RuntimeError as exc:
            self.say(str(exc))

    def disable_ai(self, _match: re.Match[str] | None = None) -> None:
        self.ai_enabled = False
        self.say("AI chat is off. Your local commands still work.")

    def reset_conversation(self, _match: re.Match[str] | None = None) -> None:
        self.previous_response_id = None
        self.say("Conversation context reset.")

    def help(self, _match: re.Match[str] | None = None) -> None:
        self.say(
            "I can tell you the time or date, calculate arithmetic, search the web, "
            "open a few websites, save notes, and (when enabled) chat with AI. "
            "Type 'quit' to leave.\n"
            "Examples: 'calculate 12 * (3 + 4)', 'search for Python tutorials', "
            "'open YouTube', 'remember buy coffee', 'show my notes', 'ai on'."
        )

    def greet(self, _match: re.Match[str] | None = None) -> None:
        self.say("Hello! I'm ready. What can I help with?")

    def tell_time(self, _match: re.Match[str] | None = None) -> None:
        self.say(f"It's {datetime.now().astimezone().strftime('%I:%M %p')}.")

    def tell_date(self, _match: re.Match[str] | None = None) -> None:
        today = datetime.now().astimezone()
        self.say(f"Today is {today.strftime('%A, %B')} {today.day}, {today.year}.")

    def do_calculate(self, match: re.Match[str]) -> None:
        expression = match.group(1).strip().rstrip("?")
        try:
            result = calculate(expression)
            self.say(f"{expression} = {result:g}" if isinstance(result, float) else f"{expression} = {result}")
        except ValueError as exc:
            self.say(str(exc))

    def search_web(self, match: re.Match[str]) -> None:
        query = match.group(1).strip()
        if not query:
            self.say("Tell me what you'd like me to search for.")
            return
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
        webbrowser.open(url)
        self.say(f"Searching the web for: {query}")

    def open_site(self, match: re.Match[str]) -> None:
        sites = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "github": "https://github.com",
            "wikipedia": "https://www.wikipedia.org",
        }
        site = match.group(1).lower()
        webbrowser.open(sites[site])
        self.say(f"Opening {site}.")

    def _read_notes(self) -> list[str]:
        try:
            notes = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
            return notes if isinstance(notes, list) and all(isinstance(n, str) for n in notes) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def add_note(self, match: re.Match[str]) -> None:
        note = match.group(1).strip()
        if not note:
            self.say("What would you like me to remember?")
            return
        notes = self._read_notes()
        notes.append(note)
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
        self.say("Got it. I've saved that note.")

    def show_notes(self, _match: re.Match[str] | None = None) -> None:
        notes = self._read_notes()
        if not notes:
            self.say("You don't have any notes yet.")
            return
        self.say("Here are your notes:\n" + "\n".join(f"{i}. {note}" for i, note in enumerate(notes, 1)))

    def clear_notes(self, _match: re.Match[str] | None = None) -> None:
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_FILE.write_text("[]", encoding="utf-8")
        self.say("Your notes are cleared.")

    def quit(self, _match: re.Match[str] | None = None) -> None:
        self.say("Goodbye!")
        self.running = False

    def handle(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        for pattern, action in self.commands:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                action(match)  # type: ignore[operator]
                return
        if self.ai_enabled:
            self._ask_ai(cleaned)
            return
        self.say("I don't know that one yet. Type 'help' to see what I can do.")

    def run(self) -> None:
        print("\nJ.A.R.V.I.S. | Personal Assistant")
        print("Type 'help' to see commands. Notes are saved on this computer.")
        if self.voice_enabled:
            print("Voice mode is active. Speak after each 'Listening...' prompt.")
        if self.ai_enabled:
            print("AI chat is active. Use 'ai off' to return to local commands.")
        while self.running:
            try:
                if self.voice_enabled:
                    print("Listening (up to 8 seconds)...")
                    text = self._listen()
                    if text:
                        print(f"You: {text}")
                        self.handle(text)
                    elif self.voice_enabled:
                        self.say("I didn't catch that. Please try again.")
                else:
                    self.handle(input("You: "))
            except (EOFError, KeyboardInterrupt):
                print()
                self.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your personal Jarvis assistant")
    parser.add_argument("--voice", action="store_true", help="Use Windows speech input and output")
    parser.add_argument("--ai", action="store_true", help="Start with AI chat enabled (requires an API key)")
    options = parser.parse_args()
    if options.voice and sys.platform != "win32":
        parser.error("--voice currently uses Windows Speech and requires Windows.")
    Jarvis(voice=options.voice, ai=options.ai).run()
