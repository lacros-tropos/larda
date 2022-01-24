#!/usr/bin/env python3

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.completion import PathCompleter


import sys, traceback
sys.path.append('../')
import datetime
# just needed to find pyLARDA from this location
# sys.path.append('../larda/')
# sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Connector as Connector

import logging
log = logging.getLogger('__main__')
log.setLevel(logging.DEBUG)
# write logs to file for debugging reasons
#log.addHandler(logging.FileHandler("promt_log", mode='w', encoding=None, delay=False))

help_text = """
Press Ctrl-f and Ctrl-b to toggle fields, ENTER to update.
Tab autocompletes the base dir and arrow keys acccess the history.
Ctrl-C will exit and provide the final output in the terminal.
"""

system_info = {
    'path': {
        'nc': 
            {'base_dir': '/lacroshome/mira/data/', 
            'matching_subdirs': '(Mom\/\d{4}.*\d{8}_\d{6}.mmclx)', 
            'date_in_filename': '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})'}},
    'generic': {}, 
    'params': {}}


def get_the_filelist(system_info):

    log.info(system_info['path']['nc']['base_dir'])
    log.info(system_info['path']['nc']['matching_subdirs'])
    log.info(system_info['path']['nc']['date_in_filename'])

    conn = Connector.Connector('empty', system_info, [["20140101", datetime.datetime.utcnow().strftime("%Y%m%d")]])
    conn.build_filehandler()

    return conn.filehandler


def format_files(d):

    s = ''

    for k,v in d.items():
        s += '{}\n'.format(k)
        s += '\n'.join([str(e) for e in v])
    return s



def main():

    output_field = TextArea(style="class:output-field", text=help_text)
    dir_field = TextArea(
        height=1,
        prompt="base_dir ",
        text=system_info['path']['nc']['base_dir'],
        style="class:input-field",
        multiline=False,
        
        completer=PathCompleter(),
        wrap_lines=False,
    )
    matching_field = TextArea(
        height=1,
        prompt="matching_subdirs ",
        text=system_info['path']['nc']['matching_subdirs'],
        style="class:input-field",
        multiline=False,
        wrap_lines=False,
    )
    date_file_field = TextArea(
        height=1,
        prompt="date_in_filename ",
        text=system_info['path']['nc']['date_in_filename'],
        style="class:input-field",
        multiline=False,
        wrap_lines=False,
    )

    container = HSplit(
        [
            dir_field,
            matching_field,
            date_file_field,
            Window(height=1, char="-", style="class:line"),
            output_field,
        ]
    )

    # The key bindings.
    kb = KeyBindings()

    kb.add("c-f")(focus_next)
    kb.add("c-b")(focus_previous)

    @kb.add("c-c")
    @kb.add("c-q")
    def _(event):
        " Pressing Ctrl-Q or Ctrl-C will exit the user interface. "
        event.app.exit()


    #@kb.add('enter')
    # Attach accept handler to the input field. We do this by assigning the
    # handler to the `TextArea` that we created earlier. it is also possible to
    # pass it to the constructor of `TextArea`.
    # NOTE: It's better to assign an `accept_handler`, rather then adding a
    #       custom ENTER key binding. This will automatically reset the input
    #       field and add the strings to the history.
    def accept(buff):
        # Evaluate "calculator" expression.
        log.info(str(buff))
        #log.info(type(buff))
        system_info['path']['nc']['base_dir'] = dir_field.text
        system_info['path']['nc']['matching_subdirs'] = matching_field.text
        system_info['path']['nc']['date_in_filename'] = date_file_field.text
        log.info(str(matching_field.__dict__))
        #log.info('matching_field.buffer ' + str(matching_field.buffer.__dict__))
        #log.info('matching_field._working_lines ' + str(matching_field.buffer._working_lines))
        #log.info('history get_strings ' + str(matching_field.buffer.history.get_strings()))


        # hack to make the history work (aka accessing old entries with arrow keys)
        # despite not resetting the field
        dir_field.buffer._working_lines.append(system_info['path']['nc']['base_dir'])
        matching_field.buffer._working_lines.append(system_info['path']['nc']['matching_subdirs'])
        date_file_field.buffer._working_lines.append(system_info['path']['nc']['date_in_filename'])
        # try:
        #     output = "\n\nIn:  {}\nOut: {}".format(
        #         input_field.text, eval(input_field.text)
        #     )  # Don't do 'eval' in real code!
        # except BaseException as e:
        #     output = "\n\n{}".format(e)

        try:
            files = get_the_filelist(system_info)
            new_text = format_files(files)
        except BaseException as e:
            tb = traceback.format_exc()
            new_text = "\n{}\n{}".format(tb, e)

        # Add text to output buffer.
        output_field.buffer.document = Document(
            text=new_text, cursor_position=len(new_text)
        )

        return True

    dir_field.accept_handler = accept
    matching_field.accept_handler = accept
    date_file_field.accept_handler = accept


    # Style.
    style = Style(
        [
            ("output-field", "bg:#000044 #ffffff"),
            ("input-field", "bg:#000000 #ffffff"),
            ("line", "#004400"),
        ]
    )

    # Run application.
    application = Application(
        layout=Layout(container, focused_element=dir_field),
        key_bindings=kb,
        style=style,
        mouse_support=True,
        full_screen=True,
    )

    application.run()
    print('')
    print('[SYSTEM.path.nc]')
    print("  base_dir = '{}'".format(system_info['path']['nc']['base_dir']))
    print("  matching_subdirs = '{}'".format(system_info['path']['nc']['matching_subdirs']))
    print("  date_in_filename = '{}'".format(system_info['path']['nc']['date_in_filename']))
    print('')


if __name__ == "__main__":
    main()
