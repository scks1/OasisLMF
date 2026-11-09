__all__ = [
    'ComputationStep',
]

import os
import pathlib
import logging
import json
import inspect
from ods_tools.oed import OedSource
from ods_tools.oed.settings import Settings, ROOT_USER_ROLE
from collections import OrderedDict

from ..utils.data import get_utctimestamp
from ..utils.exceptions import OasisException
from ..utils.inputs import update_config, str2bool, has_oasis_env, get_oasis_env, ArgumentTypeError, load_json_config
from oasislmf.utils.log import oasis_log
from ..utils.log_config import OasisLogConfig

logger = logging.getLogger(__name__)

# Roles a caller can give to the 'computation_settings' read from settings files, see
# ComputationStep.get_arguments
COMPUTATION_SETTINGS_USER_ROLES = ['admin', 'user']


class ComputationStep:
    """"Abstract" Class for all Computation Step (ExposurePreAnalysis, GulCalc, ...)
    initialise the object with all specified param un step_param and sub- ComputationStep
    provide a generic interface to get the all those parameter definitions (get_params)

    the Run method must be implemented and contain le business execution logic.
    """

    step_params = []
    chained_commands = []

    # Params shared by every Computation Step, regardless of step_params.
    # Not exposed as their own CLI flags (no 'help' key) since OasisBaseCommand
    # already registers -V/--verbose, -L/--log-level, --log-format and -C/--config
    # directly; declaring them here just makes them collectable via get_params() so
    # they can be set from a computation settings JSON file and appear in the
    # generated computation_settings_schema.json.
    #
    # 'config' is the exception: it mirrors the CLI's -C/--config MDK config file path
    # (needed so _apply_log_config() can re-read its "logging" block), not something an
    # analysis settings file should be able to set, so it's excluded from the schema.
    # 'computation_settings_user_role' is excluded for the same reason: it says how far
    # the settings files can be trusted, so they must not be able to set it themselves.
    global_params = [
        {'name': 'verbose', 'default': False},
        {'name': 'log_level', 'choices': OasisLogConfig.STANDARD_LEVELS},
        {'name': 'log_format', 'choices': list(OasisLogConfig.FORMAT_TEMPLATES.keys())},
        {'name': 'config', 'exclude_from_schema': True},
        {'name': 'computation_settings_user_role', 'default': 'admin', 'choices': COMPUTATION_SETTINGS_USER_ROLES,
         'exclude_from_schema': True},
    ]

    def __init__(self, **kwargs):
        """Initialise the ComputationStep objects:
        - do the basic check for required parameter (required)
        - provide default value if defined (default)
        - check path existence (pre_exist)
        - create necessary directories (is_dir, is_path)
        """
        self.logger = logging.getLogger(__name__)
        self.kwargs = kwargs
        self.logger.debug(f"{self.__class__.__name__}: " + json.dumps(self.kwargs, indent=4, default=str))
        self.run = oasis_log(self.run)

        # set initial value for mdk params
        for param in self.get_params():
            param_value = self._get_init_value(param, kwargs)
            setattr(self, param['name'], param_value)

        self._apply_log_config()

        # read and merge settings files
        settings = Settings()
        for settings_info in self.get_params(param_type="settings"):
            setting_fp = kwargs.get(settings_info['name'])
            if setting_fp and pathlib.Path(setting_fp).exists():
                new_settings = settings_info['loader'](setting_fp)
                settings.add_settings(new_settings, settings_info.get('user_role'))
        self.settings = settings.get_settings()

    def _get_init_value(self, param, kwargs):
        param_value = kwargs.get(param['name'])
        if param_value in [None, ""]:
            if param.get('required'):
                raise OasisException(f"parameter {param['name']} is required "
                                     f"for Computation Step {self.__class__.__name__}")
            else:
                param_value = param.get('default')

        if (getattr(param.get('type'), '__name__', None) == 'str2bool') and (not isinstance(param_value, bool)):
            try:
                param_value = str2bool(param_value)
            except ArgumentTypeError:
                raise OasisException(
                    f"The parameter '{param.get('name')}' has an invalid value '{param_value}' for boolean. Valid strings are (case insensitive):"
                    "\n  True:  ['yes', 'true', 't', 'y', '1']"
                    "\n  False: ['no', 'false', 'f', 'n', '0']"
                )

        if (param.get('is_path')
                and param_value is not None
                and not isinstance(param_value, OedSource)):
            if param.get('pre_exist') and not os.path.exists(param_value):
                raise OasisException(
                    f"The path {param_value} ({param['help']}) "
                    f"must exist for Computation Step {self.__class__.__name__}")
            else:
                if param.get('is_dir'):
                    pathlib.Path(param_value).mkdir(parents=True, exist_ok=True)
                else:
                    pathlib.Path(os.path.dirname(param_value)).mkdir(parents=True, exist_ok=True)
            param_value = str(param_value)
        return param_value

    def _apply_log_config(self):
        """Re-apply an explicitly resolved log_level/log_format to the 'oasislmf' logger.

        By the time a ComputationStep is constructed, 'log_level'/'log_format' may have
        come from a 'computation_settings' block inside an analysis settings file - which
        is only read and merged in here (via OasisComputationCommand.get_arguments), after
        the CLI already configured logging in setup_logger() using just the raw CLI args /
        MDK config file. Without this, a value set only via computation settings would be
        recorded on self.log_level/self.log_format but never actually change the logger's
        behaviour.

        Only acts when self.log_level/self.log_format are explicitly set (not None). It
        deliberately ignores self.verbose: that legacy flag is already fully and correctly
        resolved by setup_logger() (which has access to the nested MDK config "logging"
        block this class never sees), so re-deriving a level from it here - with none of
        that context - would risk silently overriding a correctly resolved level.

        self.config (the MDK config file path, set the same way as log_level/log_format)
        is re-loaded here so that a "logging" block in that file - e.g. ods_tools_level -
        is still honoured by get_ods_tools_level() when the level is re-applied.
        """
        if self.log_level is None and self.log_format is None:
            return

        config_dict = {}
        if self.config:
            try:
                config_dict = load_json_config(self.config)
            except (OasisException, json.JSONDecodeError):
                self.logger.warning(f"Could not re-load MDK config file for logging: {self.config}")

        log_config = OasisLogConfig(config_dict)
        logger = logging.getLogger('oasislmf')

        if self.log_level is not None:
            level = log_config.get_log_level(self.log_level)
            logger.setLevel(level)
            logging.getLogger('ods_tools').setLevel(log_config.get_ods_tools_level(level))

        if self.log_format is not None:
            formatter = log_config.create_formatter(self.log_format)
            for handler in logger.handlers:
                if handler.name == 'oasislmf':
                    handler.setFormatter(formatter)

    @classmethod
    def get_default_run_dir(cls):
        return os.path.join(os.getcwd(), 'runs', f'{cls.run_dir_key}-{get_utctimestamp(fmt="%Y%m%d%H%M%S")}')

    @classmethod
    def get_params(cls, param_type="step"):
        """Return all the params of the computation step defined in step_params
        and the params from the sub_computation step in chained_commands - plus,
        when param_type=="step", the params in ComputationStep.global_params
        (verbose, log_level, log_format, config).
        if two params have the same name, return the param definition of the first param found only
        this allow to overwrite the param definition of sub step if necessary.
        """
        params = {}

        def all_params():
            if param_type == "step":
                for _param in ComputationStep.global_params:
                    yield _param
            for _param in getattr(cls, f"{param_type}_params", []):
                yield _param
            for command in cls.chained_commands:
                for _param in command.get_params(param_type=param_type):
                    yield _param

        for param in all_params():
            if param['name'] not in params:
                params[param['name']] = param

        return list(params.values())

    @classmethod
    def get_arguments(cls, **kwargs):
        """Return the argument values for the step's parameters.

        Values come from the parameter defaults, then from ``kwargs`` (or ``OASIS_<NAME>``
        environment variables when ``OASIS_ENV_OVERRIDE`` is set), then from the
        ``computation_settings`` block of any settings file given in ``kwargs``.

        ``computation_settings_user_role`` (in ``kwargs``) says what that block may change:

        - ``'admin'`` (the default): it can change any parameter, including values given in
          ``kwargs``.
        - ``'user'``: it can only change parameters left at their defaults; values given in
          ``kwargs`` (other than None) or by environment override are kept. A platform can
          use this for requests from users who aren't staff.

        Parameters marked ``exclude_from_schema`` (such as ``config`` and the role itself)
        can never be set by ``computation_settings``.

        Args:
            **kwargs: argument values, by parameter name.

        Returns:
            dict: the value of each parameter, by name.
        """
        func_args = {el['name']: el.get('default', None) for el in cls.get_params()}
        type_map = {el['name']: el.get('type', None) for el in cls.get_params()}
        not_settable = {el['name'] for el in cls.get_params() if el.get('exclude_from_schema')}

        func_kwargs = update_config(kwargs)
        env_override = str2bool(os.getenv('OASIS_ENV_OVERRIDE', default=False))

        set_by_caller = set()
        for param in func_args:
            if env_override and has_oasis_env(param):
                func_args[param] = get_oasis_env(param, type_map[param])
                set_by_caller.add(param)
            elif param in func_kwargs:
                func_args[param] = func_kwargs[param]
                if func_kwargs[param] is not None:
                    set_by_caller.add(param)

        user_role = func_args.get('computation_settings_user_role') or 'admin'
        if user_role not in COMPUTATION_SETTINGS_USER_ROLES:
            raise OasisException(
                f"computation_settings_user_role must be one of {COMPUTATION_SETTINGS_USER_ROLES}, not {user_role!r}")

        computation_settings = Settings()
        if user_role == 'admin':
            computation_settings.add_settings(func_args, ROOT_USER_ROLE)
            protected = set()
        else:
            protected = set_by_caller | not_settable
            computation_settings.add_settings({k: v for k, v in func_args.items() if k not in protected})
            computation_settings.add_settings({k: v for k, v in func_args.items() if k in protected}, ROOT_USER_ROLE)

        for settings_info in cls.get_params(param_type="settings"):
            setting_fp = func_args.get(settings_info["name"])
            if setting_fp and pathlib.Path(setting_fp).exists():
                new_settings = settings_info["loader"](setting_fp)
                file_settings = new_settings.pop("computation_settings", {})
                ignored = sorted((not_settable | protected).intersection(file_settings))
                if ignored:
                    logger.warning(
                        "computation_settings in '%s': ignoring %s, which can't be set from a settings file%s",
                        setting_fp, ignored, "" if user_role == 'admin' else " by the 'user' role")
                    file_settings = {k: v for k, v in file_settings.items() if k not in ignored}
                computation_settings.add_settings(file_settings, ROOT_USER_ROLE if user_role == 'admin' else {user_role})
        return computation_settings.get_settings()

    @classmethod
    def get_signature(cls):
        """Create a function signature based on the 'get_params()' return"""
        try:
            # Create keyword params (without default values)
            params = ["{}=None".format(p.get('name')) for p in cls.get_params() if 'default' not in p]

            # Create keyword params (with default values)
            for p in [p for p in cls.get_params() if 'default' in p]:
                if isinstance(p.get('default'), str):
                    params.append("{}='{}'".format(p.get('name'), p.get('default')))
                elif isinstance(p.get('default'), dict):
                    params.append("{}=dict()".format(p.get('name'), p.get('default')))
                elif isinstance(p.get('default'), OrderedDict):
                    params.append("{}=OrderedDict()".format(p.get('name'), p.get('default')))
                else:
                    params.append("{}={}".format(p.get('name'), p.get('default')))

            namespace = {}
            exec('def func_sig({}): pass'.format(", ".join(params)), namespace)
            return inspect.signature(namespace['func_sig'])
        except Exception:
            # ignore any errors in signature creation and return blank
            return None

    @classmethod
    def get_computation_settings_json_schema(cls):
        """Return a json schema equivalent to validate the input of the command line"""
        arg_type_to_json_type = {
            str: "string",
            int: "integer",
            float: "number",
            str2bool: "boolean",
        }

        def get_json_type(_param):
            if _param.get('type') in arg_type_to_json_type:
                return arg_type_to_json_type[_param.get('type')]
            elif _param.get('is_path'):
                return "string"
            elif _param.get('default') in [True, False]:
                return "boolean"
            elif isinstance(_param.get('default'), dict):
                return "object"
            elif isinstance(_param.get('default'), list):
                return "array"
            elif isinstance(_param.get('default'), str):
                return "string"
            else:
                return "string"

        json_schema = {
            "$schema": "http://oasislmf.org/computation_settings/draft/schema#",
            "type": "object",
            "title": "Computation settings.",
            "description": "Specifies the computation settings and outputs for an analysis.",
            "additionalProperties": False,
            "properties": {}
        }
        settings_param_names = [param['name'] for param in cls.get_params(param_type="settings")]
        for param in cls.get_params():
            if param['name'] in settings_param_names:  # param is a json settings and therefore cannot be in the settings schema
                continue
            if param.get('exclude_from_schema'):  # param is CLI/MDK-config only and cannot be set via computation_settings
                continue
            param_schema = {"type": get_json_type(param)}
            if param.get('help'):
                param_schema["description"] = param['help']
            if param.get('choices'):
                param_schema["enum"] = param.get('choices')
            json_schema["properties"][param['name']] = param_schema
        return json_schema

    def run(self):
        """Method that will be call by all the interface to execute the computation step"""
        raise NotImplementedError(f'Method run must be implemented in {self.__class__.__name__}')
