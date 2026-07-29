"""Keep Condor bot-menu callback_data below Telegram's 64-byte limit."""

from pathlib import Path


menu = Path("/opt/condor/handlers/bots/menu.py")
source = menu.read_text(encoding="utf-8")
source = source.replace("for bot_name in bot_names:\n            display_name", "for index, bot_name in enumerate(bot_names):\n            display_name", 1)
source = source.replace('callback_data=f"bots:bot_detail:{bot_name}",', 'callback_data=f"bots:bot_detail_idx:{index}",', 1)
source = source.replace('callback_data=f"bots:bot_detail:{bot_name}",', 'callback_data=f"bots:bot_detail_idx:{j}",', 1)
needle = '        context.user_data["active_bots_data"] = bots_data\n'
replacement = needle + '        context.user_data["active_bot_names"] = list(bots_dict.keys())\n'
if needle not in source or "bots:bot_detail_idx" not in source:
    raise SystemExit("Condor bot-menu source changed; review callback patch.")
menu.write_text(source.replace(needle, replacement, 1), encoding="utf-8")

handlers = Path("/opt/condor/handlers/bots/__init__.py")
source = handlers.read_text(encoding="utf-8")
needle = '''        # Bot detail
        elif main_action == "bot_detail":
            if len(action_parts) > 1:
                bot_name = action_parts[1]
                await show_bot_detail(update, context, bot_name)
'''
replacement = '''        # Bot detail. Telegram callback_data is limited to 64 bytes, so
        # long API-created instance names use a short per-menu index.
        elif main_action == "bot_detail_idx":
            if len(action_parts) > 1:
                try:
                    bot_name = context.user_data.get("active_bot_names", [])[int(action_parts[1])]
                except (IndexError, ValueError):
                    await show_bots_menu(update, context)
                else:
                    await show_bot_detail(update, context, bot_name)
        elif main_action == "bot_detail":
            if len(action_parts) > 1:
                bot_name = action_parts[1]
                await show_bot_detail(update, context, bot_name)
'''
if needle not in source:
    raise SystemExit("Condor bot handler source changed; review callback patch.")
handlers.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
