from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

# these tables and documenation are extracted from
# https://www.microsoft.com/typography/otspec/name.htm

MAC_LANG_TABLE = {0: 'English',
                  1: 'French',
                  2: 'German',
                  3: 'Italian',
                  4: 'Dutch',
                  5: 'Swedish',
                  6: 'Spanish',
                  7: 'Danish',
                  8: 'Portuguese',
                  9: 'Norwegian',
                  10: 'Hebrew',
                  11: 'Japanese',
                  12: 'Arabic',
                  13: 'Finnish',
                  14: 'Inuktitut',
                  15: 'Icelandic',
                  16: 'Maltese',
                  17: 'Turkish',
                  18: 'Croatian',
                  19: 'Chinese (Traditional)',
                  20: 'Urdu',
                  21: 'Hindi',
                  22: 'Thai',
                  23: 'Korean',
                  24: 'Lithuanian',
                  25: 'Polish',
                  26: 'Hungarian',
                  27: 'Estonian',
                  28: 'Latvian',
                  29: 'Sami',
                  30: 'Faroese',
                  31: 'Farsi/Persian',
                  32: 'Russian',
                  33: 'Chinese (Simplified)',
                  34: 'Flemish',
                  35: 'Irish Gaelic',
                  36: 'Albanian',
                  37: 'Romanian',
                  38: 'Czech',
                  39: 'Slovak',
                  40: 'Slovenian',
                  41: 'Yiddish',
                  42: 'Serbian',
                  43: 'Macedonian',
                  44: 'Bulgarian',
                  45: 'Ukrainian',
                  46: 'Byelorussian',
                  47: 'Uzbek',
                  48: 'Kazakh',
                  49: 'Azerbaijani (Cyrillic script)',
                  50: 'Azerbaijani (Arabic script)',
                  51: 'Armenian',
                  52: 'Georgian',
                  53: 'Moldavian',
                  54: 'Kirghiz',
                  55: 'Tajiki',
                  56: 'Turkmen',
                  57: 'Mongolian (Mongolian script)',
                  58: 'Mongolian (Cyrillic script)',
                  59: 'Pashto',
                  60: 'Kurdish',
                  61: 'Kashmiri',
                  62: 'Sindhi',
                  63: 'Tibetan',
                  64: 'Nepali',
                  65: 'Sanskrit',
                  66: 'Marathi',
                  67: 'Bengali',
                  68: 'Assamese',
                  69: 'Gujarati',
                  70: 'Punjabi',
                  71: 'Oriya',
                  72: 'Malayalam',
                  73: 'Kannada',
                  74: 'Tamil',
                  75: 'Telugu',
                  76: 'Sinhalese',
                  77: 'Burmese',
                  78: 'Khmer',
                  79: 'Lao',
                  80: 'Vietnamese',
                  81: 'Indonesian',
                  82: 'Tagalong',
                  83: 'Malay (Roman script)',
                  84: 'Malay (Arabic script)',
                  85: 'Amharic',
                  86: 'Tigrinya',
                  87: 'Galla',
                  88: 'Somali',
                  89: 'Swahili',
                  90: 'Kinyarwanda/Ruanda',
                  91: 'Rundi',
                  92: 'Nyanja/Chewa',
                  93: 'Malagasy',
                  94: 'Esperanto',
                  128: 'Welsh',
                  129: 'Basque',
                  130: 'Catalan',
                  131: 'Latin',
                  132: 'Quenchua',
                  133: 'Guarani',
                  134: 'Aymara',
                  135: 'Tatar',
                  136: 'Uighur',
                  137: 'Dzongkha',
                  138: 'Javanese (Roman script)',
                  139: 'Sundanese (Roman script)',
                  140: 'Galician',
                  141: 'Afrikaans',
                  142: 'Breton',
                  144: 'Scottish Gaelic',
                  145: 'Manx Gaelic',
                  146: 'Irish Gaelic (with dot above)',
                  147: 'Tongan',
                  148: 'Greek (polytonic)',
                  149: 'Greenlandic',
                  150: 'Azerbaijani (Roman script)'}


NAME_ID_TABLE = {
    0: ("copyright", """Copyright notice."""),
    1: ('family_name',
        """Font Family name. Up to four fonts can share the Font Family name,
forming a font style linking group (regular, italic, bold, bold italic
— as defined by OS/2.fsSelection bit settings)."""),
    2: ('sub_family',
        """Font Subfamily name. The Font Subfamily name distiguishes the font
in a group with the same Font Family name (name ID 1). This is assumed
to address style (italic, oblique) and weight (light, bold, black,
etc.). A font with no particular differences in weight or style
(e.g. medium weight, not italic and fsSelection bit 6 set) should have
the string “Regular” stored in this position."""),
    3: ('ufi', """Unique font identifier"""),
    4: ('full_font_name',
        """Full font name; a combination of strings 1 and 2, or a similar
human-readable variant. If string 2 is "Regular", it is sometimes
omitted from name ID 4."""),
    5: ('version',
        """Version string. Should begin with the syntax 'Version
<number>.<number>' (upper case, lower case, or mixed, with a space
between “Version” and the number).

The string must contain a version number of the following form: one or
more digits (0-9) of value less than 65,535, followed by a period,
followed by one or more digits of value less than 65,535. Any
character other than a digit will terminate the minor number. A
character such as “;” is helpful to separate different pieces of
version information.

The first such match in the string can be used by installation
software to compare font versions. Note that some installers may
require the string to start with “Version ”, followed by a version
number as above."""),
    6: ('ps_name',
        """Postscript name for the font; Name ID 6 specifies a string which is
used to invoke a PostScript language font that corresponds to this
OpenType font. When translated to ASCII, the name string must be no
longer than 63 characters and restricted to the printable ASCII
subset, codes 33 to 126, except for the 10 characters '[', ']', '(',
')', '{', '}', '<', '>', '/', '%'.

In a CFF OpenType font, there is no requirement that this name be the
same as the font name in the CFF’s Name INDEX. Thus, the same CFF may
be shared among multiple font components in a Font Collection. See the
'name' table section of Recommendations for OpenType fonts "" for
additional information."""),
    7: ('trademark',
        """Trademark; this is used to save any trademark notice/information
for this font. Such information should be based on legal advice. This
is distinctly separate from the copyright."""),
    8: ('manufacturer', """Manufacturer Name."""),
    9: ('designer', """Designer; name of the designer of the typeface."""),
    10: ('description',
         """Description; description of the typeface. Can contain revision
information, usage recommendations, history, features, etc."""),
    11: ('url_vendor',
         """URL Vendor; URL of font vendor (with protocol, e.g., http://,
ftp://). If a unique serial number is embedded in the URL, it can be
used to register the font."""),
    12: ('url_designer',
         """URL Designer; URL of typeface designer (with protocol, e.g.,
http://, ftp://)."""),
    13: ('llicense_desc',
         """License Description; description of how the font may be legally
used, or different example scenarios for licensed use. This field
should be written in plain language, not legalese."""),
    14: ('license_url',
         """License Info URL; URL where additional licensing information can be
found."""),
    15: ('Reserved.', ''),
    16: ('typographic_family_name',
         """Typographic Family name: The typographic family grouping doesn't
impose any constraints on the number of faces within it, in contrast
with the 4-style family grouping (ID 1), which is present both for
historical reasons and to express style linking groups. If name ID 16
is absent, then name ID 1 is considered to be the typographic family
name. (In earlier versions of the specification, name ID 16 was known
as "Preferred Family".)"""),
    17: ('typographic_sub_family',
         """Typographic Subfamily name: This allows font designers to specify a
subfamily name within the typographic family grouping. This string
must be unique within a particular typographic family. If it is
absent, then name ID 2 is considered to be the typographic subfamily
name. (In earlier versions of the specification, name ID 17 was known
as "Preferred Subfamily".)"""),
    18: ('compat_ful_font_name',
         """Compatible Full (Macintosh only); On the Macintosh, the menu name
is constructed using the FOND resource. This usually matches the Full
Name. If you want the name of the font to appear differently than the
Full Name, you can insert the Compatible Full Name in ID 18."""),
    19: ('sample_text',
         """Sample text; This can be the font name, or any other text that the
designer thinks is the best sample to display the font in."""),
    20: ('ps_cid',
         """PostScript CID findfont name; Its presence in a font means that the
nameID 6 holds a PostScript font name that is meant to be used with
the “composefont” invocation in order to invoke the font in a
PostScript interpreter. See the definition of name ID 6.

The value held in the name ID 20 string is interpreted as a PostScript
font name that is meant to be used with the “findfont” invocation, in
order to invoke the font in a PostScript interpreter.  When translated
to ASCII, this name string must be restricted to the printable ASCII
subset, codes 33 through 126, except for the 10 characters: '[', ']',
'(', ')', '{', '}', '<', '>', '/', '%'.

See "Recommendations for OTF fonts" for additional information"""),
    21: ('WWS_family_name',
         """WWS Family Name. Used to provide a WWS-conformant family name in
case the entries for IDs 16 and 17 do not conform to the WWS
model. (That is, in case the entry for ID 17 includes qualifiers for
some attribute other than weight, width or slope.) If bit 8 of the
fsSelection field is set, a WWS Family Name entry should not be needed
and should not be included. Conversely, if an entry for this ID is
include, bit 8 should not be set. (See OS/2 'fsSelection' field for
details.) Examples of name ID 21: “Minion Pro Caption” and “Minion Pro
Display”. (Name ID 16 would be “Minion Pro” for these examples.)"""),
    22: ('WWS_sub_family',
         """WWS Subfamily Name. Used in conjunction with ID 21, this ID
provides a WWS-conformant subfamily name (reflecting only weight,
width and slope attributes) in case the entries for IDs 16 and 17 do
not conform to the WWS model. As in the case of ID 21, use of this ID
should correlate inversely with bit 8 of the fsSelection field being
set. Examples of name ID 22: “Semibold Italic”, “Bold
Condensed”. (Name ID 17 could be “Semibold Italic Caption”, or “Bold
Condensed Display”, for example.) """),
    23: ('lbp',
         """Light Backgound Palette. This ID, if used in the CPAL table’s
Palette Labels Array, specifies that the corresponding color palette
in the CPAL table is appropriate to use with the font when displaying
it on a light background such as white. Name table strings for this ID
specify the user interface strings associated with this palette."""),
    24: ('dbp',
         """Dark Backgound Palette. This ID, if used in the CPAL table’s
Palette Labels Array, specifies that the corresponding color palette
in the CPAL table is appropriate to use with the font when displaying
it on a dark background such as black. Name table strings for this ID
specify the user interface strings associated with this palette"""),
    25: ('var_ps_name',
         """Variations PostScript Name Prefix. If present in a variable font,
it may be used as the family prefix in the PostScript Name Generation
for Variation Fonts algorithm. The character set is restricted to
ASCII-range uppercase Latin letters, lowercase Latin letters, and
digits. All name strings for name ID 25 within a font, when converted
to ASCII, must be identical. See Adobe Technical Note #5902:
“PostScript Name Generation for Variation Fonts” for reasons to
include name ID 25 in a font, and for examples. For general
information on OpenType Font Variations, see the chapter, OpenType
Font Variations Overview.""")}

WINDOWS_LANG_TABLE = {
    1025: 'Arabic (Saudi Arabia)',
    1026: 'Bulgarian (Bulgaria)',
    1027: 'Catalan (Catalan)',
    1028: 'Chinese (Taiwan)',
    1029: 'Czech (Czech Republic)',
    1030: 'Danish (Denmark)',
    1031: 'German (Germany)',
    1032: 'Greek (Greece)',
    1033: 'English (United States)',
    1034: 'Spanish (Traditional Sort) (Spain)',
    1035: 'Finnish (Finland)',
    1036: 'French (France)',
    1037: 'Hebrew (Israel)',
    1038: 'Hungarian (Hungary)',
    1039: 'Icelandic (Iceland)',
    1040: 'Italian (Italy)',
    1041: 'Japanese (Japan)',
    1042: 'Korean (Korea)',
    1043: 'Dutch (Netherlands)',
    1044: 'Norwegian (Bokmal) (Norway)',
    1045: 'Polish (Poland)',
    1046: 'Portuguese (Brazil)',
    1047: 'Romansh (Switzerland)',
    1048: 'Romanian (Romania)',
    1049: 'Russian (Russia)',
    1050: 'Croatian (Croatia)',
    1051: 'Slovak (Slovakia)',
    1052: 'Albanian (Albania)',
    1053: 'Swedish (Sweden)',
    1054: 'Thai (Thailand)',
    1055: 'Turkish (Turkey)',
    1056: 'Urdu (Islamic Republic of Pakistan)',
    1057: 'Indonesian (Indonesia)',
    1058: 'Ukrainian (Ukraine)',
    1059: 'Belarusian (Belarus)',
    1060: 'Slovenian (Slovenia)',
    1061: 'Estonian (Estonia)',
    1062: 'Latvian (Latvia)',
    1063: 'Lithuanian (Lithuania)',
    1064: 'Tajik (Cyrillic) (Tajikistan)',
    1066: 'Vietnamese (Vietnam)',
    1067: 'Armenian (Armenia)',
    1068: 'Azeri (Latin) (Azerbaijan)',
    1069: 'Basque (Basque)',
    1070: 'Upper Sorbian (Germany)',
    1071: 'Macedonian (FYROM) (Former Yugoslav Republic of Macedonia)',
    1074: 'Setswana (South Africa)',
    1076: 'isiXhosa (South Africa)',
    1077: 'isiZulu (South Africa)',
    1078: 'Afrikaans (South Africa)',
    1079: 'Georgian (Georgia)',
    1080: 'Faroese (Faroe Islands)',
    1081: 'Hindi (India)',
    1082: 'Maltese (Malta)',
    1083: 'Sami (Northern) (Norway)',
    1086: 'Malay (Malaysia)',
    1087: 'Kazakh (Kazakhstan)',
    1088: 'Kyrgyz (Kyrgyzstan)',
    1089: 'Kiswahili (Kenya)',
    1090: 'Turkmen (Turkmenistan)',
    1091: 'Uzbek (Latin) (Uzbekistan)',
    1092: 'Tatar (Russia)',
    1093: 'Bengali (India)',
    1094: 'Punjabi (India)',
    1095: 'Gujarati (India)',
    1096: 'Odia (formerly Oriya) (India)',
    1097: 'Tamil (India)',
    1098: 'Telugu (India)',
    1099: 'Kannada (India)',
    1100: 'Malayalam (India)',
    1101: 'Assamese (India)',
    1102: 'Marathi (India)',
    1103: 'Sanskrit (India)',
    1104: 'Mongolian (Cyrillic) (Mongolia)',
    1105: 'Tibetan (PRC)',
    1106: 'Welsh (United Kingdom)',
    1107: 'Khmer (Cambodia)',
    1108: 'Lao (Lao P.D.R.)',
    1110: 'Galician (Galician)',
    1111: 'Konkani (India)',
    1114: 'Syriac (Syria)',
    1115: 'Sinhala (Sri Lanka)',
    1117: 'Inuktitut (Canada)',
    1118: 'Amharic (Ethiopia)',
    1121: 'Nepali (Nepal)',
    1122: 'Frisian (Netherlands)',
    1123: 'Pashto (Afghanistan)',
    1124: 'Filipino (Philippines)',
    1125: 'Divehi (Maldives)',
    1128: 'Hausa (Latin) (Nigeria)',
    1130: 'Yoruba (Nigeria)',
    1131: 'Quechua (Bolivia)',
    1132: 'Sesotho sa Leboa (South Africa)',
    1133: 'Bashkir (Russia)',
    1134: 'Luxembourgish (Luxembourg)',
    1135: 'Greenlandic (Greenland)',
    1136: 'Igbo (Nigeria)',
    1144: 'Yi (PRC)',
    1146: 'Mapudungun (Chile)',
    1148: 'Mohawk (Mohawk)',
    1150: 'Breton (France)',
    1152: 'Uighur (PRC)',
    1153: 'Maori (New Zealand)',
    1154: 'Occitan (France)',
    1155: 'Corsican (France)',
    1156: 'Alsatian (France)',
    1157: 'Yakut (Russia)',
    1158: "K'iche (Guatemala)",
    1159: 'Kinyarwanda (Rwanda)',
    1160: 'Wolof (Senegal)',
    1164: 'Dari (Afghanistan)',
    2049: 'Arabic (Iraq)',
    2052: "Chinese (People's Republic of China)",
    2055: 'German (Switzerland)',
    2057: 'English (United Kingdom)',
    2058: 'Spanish (Mexico)',
    2060: 'French (Belgium)',
    2064: 'Italian (Switzerland)',
    2067: 'Dutch (Belgium)',
    2068: 'Norwegian (Nynorsk) (Norway)',
    2070: 'Portuguese (Portugal)',
    2074: 'Serbian (Latin) (Serbia)',
    2077: 'Sweden (Finland)',
    2092: 'Azeri (Cyrillic) (Azerbaijan)',
    2094: 'Lower Sorbian (Germany)',
    2107: 'Sami (Northern) (Sweden)',
    2108: 'Irish (Ireland)',
    2110: 'Malay (Brunei Darussalam)',
    2115: 'Uzbek (Cyrillic) (Uzbekistan)',
    2117: 'Bengali (Bangladesh)',
    2128: "Mongolian (Traditional) (People's Republic of China)",
    2141: 'Inuktitut (Latin) (Canada)',
    2143: 'Tamazight (Latin) (Algeria)',
    2155: 'Quechua (Ecuador)',
    3073: 'Arabic (Egypt)',
    3076: 'Chinese (Hong Kong S.A.R.)',
    3079: 'German (Austria)',
    3081: 'English (Australia)',
    3082: 'Spanish (Modern Sort) (Spain)',
    3084: 'French (Canada)',
    3098: 'Serbian (Cyrillic) (Serbia)',
    3131: 'Sami (Northern) (Finland)',
    3179: 'Quechua (Peru)',
    4097: 'Arabic (Libya)',
    4100: 'Chinese (Singapore)',
    4103: 'German (Luxembourg)',
    4105: 'English (Canada)',
    4106: 'Spanish (Guatemala)',
    4108: 'French (Switzerland)',
    4122: 'Croatian (Latin) (Bosnia and Herzegovina)',
    4155: 'Sami (Lule) (Norway)',
    5121: 'Arabic (Algeria)',
    5124: 'Chinese (Macao S.A.R.)',
    5127: 'German (Liechtenstein)',
    5129: 'English (New Zealand)',
    5130: 'Spanish (Costa Rica)',
    5132: 'French (Luxembourg)',
    5146: 'Bosnian (Latin) (Bosnia and Herzegovina)',
    5179: 'Sami (Lule) (Sweden)',
    6145: 'Arabic (Morocco)',
    6153: 'English (Ireland)',
    6154: 'Spanish (Panama)',
    6156: 'French (Principality of Monoco)',
    6170: 'Serbian (Latin) (Bosnia and Herzegovina)',
    6203: 'Sami (Southern) (Norway)',
    7169: 'Arabic (Tunisia)',
    7177: 'English (South Africa)',
    7178: 'Spanish (Dominican Republic)',
    7194: 'Serbian (Cyrillic) (Bosnia and Herzegovina)',
    7227: 'Sami (Southern) (Sweden)',
    8193: 'Arabic (Oman)',
    8201: 'English (Jamaica)',
    8202: 'Spanish (Venezuela)',
    8218: 'Bosnian (Cyrillic) (Bosnia and Herzegovina)',
    8251: 'Sami (Skolt) (Finland)',
    9217: 'Arabic (Yemen)',
    9225: 'English (Caribbean)',
    9226: 'Spanish (Colombia)',
    9275: 'Sami (Inari) (Finland)',
    10241: 'Arabic (Syria)',
    10249: 'English (Belize)',
    10250: 'Spanish (Peru)',
    11265: 'Arabic (Jordan)',
    11273: 'English (Trinidad and Tobago)',
    11274: 'Spanish (Argentina)',
    12289: 'Arabic (Lebanon)',
    12297: 'English (Zimbabwe)',
    12298: 'Spanish (Ecuador)',
    13313: 'Arabic (Kuwait)',
    13321: 'English (Republic of the Philippines)',
    13322: 'Spanish (Chile)',
    14337: 'Arabic (U.A.E.)',
    14346: 'Spanish (Uruguay)',
    15361: 'Arabic (Bahrain)',
    15370: 'Spanish (Paraguay)',
    16385: 'Arabic (Qatar)',
    16393: 'English (India)',
    16394: 'Spanish (Bolivia)',
    17417: 'English (Malaysia)',
    17418: 'Spanish (El Salvador)',
    18441: 'English (Singapore)',
    18442: 'Spanish (Honduras)',
    19466: 'Spanish (Nicaragua)',
    20490: 'Spanish (Puerto Rico)',
    21514: 'Spanish (United States)'
}


def decode_name_table(name_table, strict=False):
    """Decode the contents of the name table from otf/ttf files

    The 'name' table in otf/ttf headers contains a metadata about the
    font including it's family name, sub family, information about the
    font designer, manufacturer, and vendor and legal information.  This
    information is stored as bytes with variable encoding (which is
    specified as part of the table).

    A given file may contain the same information in multiple encodings
    and languages.

    The dictionary return by  `FT2Font.get_sfnt()` has the keys::

        (platform_id, encoding, language_code, name_id)

    The ``platform_id`` controls how the integers in ``encoding`` and
    ``langague_code`` are used to determine how to decode and
    interpret the bytes in the payload.

    This function consumes the first two entries in this key, decodes
    the value and maps the name_id and language code to strings.

    Does not support language tags.

    Parameters
    ----------
    name_table : dict
        Dict[Tuple[int, int, int, int], Bytes]

    strict : bool, optional
        If `True` raise on unsupported or invalid keys.  If `False` (default)
        silently skip.

    Yields
    ------
    key : tuple
        Tuple[str, str]  of (name_id, language_code).

    v : str
        The value of the table entry.

    """
    for (pid, enc_id, lang_id, name_id), v in name_table.items():
        if name_id > 255:
            if strict:
                raise ValueError('Received a name_id index of {} which is '
                                 'greater than 255 and not '
                                 'allowed'.format(name_id))
            else:
                continue
        if pid not in (0, 1, 3):
            # the spec defines 0, 1, 2, 3 and reserves 240-255 for
            # user defined platforms
            if strict:
                raise ValueError('Received a platform_id of {} which is '
                                 'not supported. 2 is deprecated and '
                                 ' > 3 is not in the spec'.format(pid))
            else:
                continue
        # mac encoding
        if pid == 1:
            if enc_id != 0:
                if strict:
                    raise ValueError('For the mac platform received an '
                                     'encoding of {} which not 0 (Roman) '
                                     'currently the only supported mac '
                                     'encoding by Matplotlib'.format(enc_id))
                else:
                    continue
            #
            yield ((NAME_ID_TABLE.get(name_id, (name_id,))[0],
                   MAC_LANG_TABLE[lang_id]),
                   v.decode('macroman'))
        # MS encoding
        elif pid == 3:
            if enc_id not in (1, 3):
                continue
            yield ((NAME_ID_TABLE.get(name_id, (str(name_id),))[0],
                   WINDOWS_LANG_TABLE.get(
                       lang_id, '0x{:04x}'.format(lang_id))),
                   v.decode('utf_16_be'))
        # unicode
        elif pid == 0:
            # This may not always work and does not actually consult the
            # encoding.
            yield ((NAME_ID_TABLE.get(name_id, (str(name_id),))[0], ''),
                   v.decode('utf_16_be'))


def get_all_style_strings(header_dict):
    for (name, lang), v in decode_name_table(header_dict):
        # all of the places something related to the weight / style /
        # whatever might be hiding
        if name in {'typographic_sub_family',
                    'sub_family'}:
            yield v
