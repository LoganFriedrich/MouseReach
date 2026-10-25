"""Keys in the review windows: what you type stays in the panel, and the
video's keys are the review tool's keys.

WHY (Logan, 2026-10-24: "when I type numbers into text fields in the widgets
... sometimes keybinds go off"). Two things sent typed keys to the wrong place:

1. **A key a panel control does not use went to the video.** napari's dock
   hands every key the docked panel leaves unused to the viewer's shortcuts
   (``QtViewerDockWidget.keyPressEvent``). A number box keeps the digits typed
   into it while it has the keyboard, but the keyboard is usually on whatever
   was clicked last -- the "enter frames manually" / "No, it is at frame:"
   choice, a "Use current frame" button, the outcome list -- so the frame number
   typed next ran shortcuts instead. ``keep_typing_in_panel`` stops typing keys
   (letters, digits, punctuation, Backspace, Delete, Enter) there. Space and
   the arrow keys still reach the video, as before: a panel control that does
   not use them passes them on, so stepping after picking an outcome works.

2. **The video's keys were napari's layer-editing shortcuts.** napari looks a
   key up in the SELECTED layer's shortcuts before the viewer's, and the
   selected layer was the tracking dots (added last): 1 deleted selected dots,
   2-5 switched the dots to add / select / pan / transform mode (in add mode a
   click on the video adds a dot), Delete and Backspace deleted dots, A
   selected all of them; on the video layer 2 made a drag move the picture. So
   the review tool's own 1-6 speed keys never ran. A review window edits
   nothing, so ``review_keys_on_layer`` switches those off on its layers and
   binds the review tool's keys there instead.

The rule for a reviewer: the video's keys work while the video has the
keyboard -- click the video to give it the keyboard.
"""
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QDockWidget

# Editing keys a stray key press must never turn into a shortcut, with or
# without modifiers: Ctrl+Backspace / Ctrl+Delete are napari's "delete the
# selected layer" (the video or the dots).
_EDITING_KEYS = (Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Return, Qt.Key_Enter)


def is_typing_key(key, modifiers, text) -> bool:
    """True for a key that types something: a printable character other than a
    space (no Ctrl/Alt/Meta held), or Backspace, Delete, Enter."""
    if key in _EDITING_KEYS:
        return True
    if modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
        return False
    return bool(text) and text != " " and text.isprintable()


class _TypingStaysInPanel(QObject):
    """Event filter on the dock that holds a review panel. A key reaches the
    dock only when the control that has the keyboard left it unused; a typing
    key stops there instead of going on to the video's shortcuts."""

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.KeyPress, QEvent.KeyRelease) and is_typing_key(
                event.key(), event.modifiers(), event.text()):
            event.accept()
            return True
        return False


def keep_typing_in_panel(widget) -> bool:
    """Install the typing guard on the napari dock that holds ``widget``.

    Call it once the widget is docked (the review widgets do it from their
    showEvent); calling it again is harmless. Returns False when ``widget`` is
    not inside a dock (nothing to guard)."""
    dock = widget.parentWidget()
    while dock is not None and not isinstance(dock, QDockWidget):
        dock = dock.parentWidget()
    if dock is None:
        return False
    if getattr(dock, "_mousereach_typing_guard", None) is None:
        guard = _TypingStaysInPanel(dock)
        dock.installEventFilter(guard)
        dock._mousereach_typing_guard = guard
    return True


def _switched_off(layer):
    """A napari layer shortcut, switched off in a review window."""


def review_keys_on_layer(layer, review_keys) -> None:
    """Make ``layer`` answer the review tool's keys and none of napari's own.

    ``review_keys`` maps a key ("Space", "Shift-Left", "3", ...) to a function
    taking one argument (it is passed the layer and ignores it). napari's
    shortcuts for the layer's kind are read from its class keymaps, which napari
    fills when the viewer is created -- so this is called after the layer is
    added. Bindings stay with the layer object, so a layer that is reused for the
    next video keeps them."""
    napari_keys = set()
    for cls in type(layer).__mro__:
        napari_keys.update(getattr(cls, "class_keymap", None) or {})
    for key in napari_keys:
        layer.bind_key(key, _switched_off, overwrite=True)
    for key, func in review_keys.items():
        layer.bind_key(key, func, overwrite=True)
