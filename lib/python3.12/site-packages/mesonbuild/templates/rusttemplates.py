# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 The Meson development team

from __future__ import annotations

import typing as T

from mesonbuild.templates.sampleimpl import FileImpl


lib_rust_template = '''#![crate_name = "{crate_file}"]

/* This function will not be exported and is not
 * directly callable by users of this library.
 */
fn internal_function() -> i32 {{
    return 0;
}}

pub fn {function_name}() -> i32 {{
    return internal_function();
}}
'''

lib_rust_test_template = '''extern crate {crate_file};

fn main() {{
    println!("printing: {{}}", {crate_file}::{function_name}());
}}
'''


lib_rust_meson_template = '''project('{project_name}', 'rust',
  version : '{version}',
  default_options : ['warning_level=3'])

shlib = static_library('{lib_name}', '{source_file}', install : true)

test_exe = executable('{test_exe_name}', '{test_source_file}',
  link_with : shlib)
test('{test_name}', test_exe)

# Make this library usable as a Meson subproject.
{ltoken}_dep = declare_dependency(
  include_directories: include_directories('.'),
  link_with : shlib)
'''

hello_rust_template = '''
fn main() {{
    let project_name = "{project_name}";
    println!("This is project {{}}.\\n", project_name);
}}
'''

hello_rust_meson_template = '''project('{project_name}', 'rust',
  version : '{version}',
  default_options : ['warning_level=3'])

exe = executable('{exe_name}', '{source_name}',
  install : true)

test('basic', exe)
'''


class RustProject(FileImpl):

    source_ext = 'rs'
    exe_template = hello_rust_template
    exe_meson_template = hello_rust_meson_template
    lib_template = lib_rust_template
    lib_test_template = lib_rust_test_template
    lib_meson_template = lib_rust_meson_template

    def lib_kwargs(self) -> T.Dict[str, str]:
        kwargs = super().lib_kwargs()
        kwargs['crate_file'] = self.lowercase_token
        return kwargs
