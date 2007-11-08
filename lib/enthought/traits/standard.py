#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: David C. Morrill
# Date: 07/28/2003
#------------------------------------------------------------------------------
""" Defines a set of standard, commonly useful predefined traits.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.trait_handlers import TraitString, TraitPrefixList, \
                                            TraitEnum, TraitList
from enthought.traits.api    import Trait, Event

#-------------------------------------------------------------------------------
#  Trait Editor definitions:
#  Attempt to import traits.ui...on failure, set editors to None:
#-------------------------------------------------------------------------------

try:
    from enthought.traits.ui.api import editors
    boolean_editor = editors.BooleanEditor
    button_editor  = editors.ButtonEditor
except ImportError:
    boolean_editor = None
    button_editor  = None

#-------------------------------------------------------------------------------
#  Event traits:
#-------------------------------------------------------------------------------

# Event trait with button editor
event_trait = Event( editor = button_editor )

#-------------------------------------------------------------------------------
#  Boolean traits: 
#-------------------------------------------------------------------------------

# Boolean that defaults to False and uses BooleanEditor.
false_trait = Trait( False, bool, editor = boolean_editor )
# Boolean that defaults to True and uses BooleanEditor
true_trait  = Trait( True,  bool, editor = boolean_editor )

# Boolean that defaults to True, uses a BooleanEditor, and accepts a variety
# of obvious values for True and False.
flexible_true_trait = Trait( 'true',  
                         { 'true':  1, 't': 1, 'yes': 1, 'y': 1, 'on':  1, 1: 1,
                           'false': 0, 'f': 0, 'no':  0, 'n': 0, 'off': 0, 0: 0
                         }, editor = boolean_editor )
# Boolean that defaults to False, uses a BooleanEditor, and accepts a variety
# of obvious values for True and False.
flexible_false_trait = Trait( 'false', flexible_true_trait )

#-------------------------------------------------------------------------------
#  Zip Code related traits:
#-------------------------------------------------------------------------------

# Five-digit US zip code (DDDDD)
zipcode_5_trait = Trait( '99999', 
                         TraitString( regex = r'^\d{5,5}$' ) )
                   
# Nine-digit US zip code (DDDDD-DDDD)          
zipcode_9_trait = Trait( '99999-9999', 
                         TraitString( regex = r'^\d{5,5}[ -]?\d{4,4}$' ) )

#-------------------------------------------------------------------------------
#  United States state related traits:
#-------------------------------------------------------------------------------

# Long form of the 50 United States state names
us_states_long_trait = Trait( 'Texas', TraitPrefixList( [                        
   'Alabama',        'Alaska',       'Arizona',      'Arkansas',      
   'California',     'Colorado',     'Connecticut',  'Delaware',
   'Florida',        'Georgia',      'Hawaii',       'Idaho',       
   'Illinois',       'Indiana',      'Iowa',         'Kansas', 
   'Kentucky',       'Louisiana',    'Maine',        'Maryland', 
   'Massachusetts',  'Michigan',     'Minnesota',    'Mississippi',   
   'Missouri',       'Montana',      'Nebraska',     'Nevada',
   'New Hampshire',  'New Jersey',   'New Mexico',   'New York',   
   'North Carolina', 'North Dakota', 'Ohio',         'Oklahoma', 
   'Oregon',         'Pennsylvania', 'Rhode Island', 'South Carolina',
   'South Dakota',   'Tennessee',    'Texas',        'Utah', 
   'Vermont',        'Virginia',     'Washington',   'West Virginia',  
   'Wisconsin',      'Wyoming' ] ) )
   
# Postal abbreviations of the 50 United States state names
us_states_short_trait = Trait( 'TX', [   
   'AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
   'HI', 'ID', 'IA', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
   'ME', 'MI', 'MO', 'MN', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH',
   'NJ', 'NM', 'NY', 'NV', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
   'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY' ] )
                    
# Long form of all United States state and territory names
all_us_states_long_trait = Trait( 'Texas', TraitPrefixList( [                    
   'Alabama',       'Alaska',              'American Samoa', 'Arizona',
   'Arkansas',      'California',          'Colorado',       'Connecticut',
   'Delaware',      'District of Columbia','Florida',        'Georgia',
   'Guam',          'Hawaii',              'Idaho',          'Illinois',
   'Indiana',       'Iowa',                'Kansas',         'Kentucky',
   'Louisiana',     'Maine',               'Maryland',       'Massachusetts',
   'Michigan',      'Minnesota',           'Mississippi',    'Missouri',
   'Montana',       'Nebraska',            'Nevada',         'New Hampshire',
   'New Jersey',    'New Mexico',          'New York',       'North Carolina',
   'North Dakota',  'Ohio',                'Oklahoma',       'Oregon',
   'Pennsylvania',  'Puerto Rico',         'Rhode Island',   'South Carolina',
   'South Dakota',  'Tennessee',           'Texas',          'Utah',
   'Vermont',       'Virgin Islands',      'Virginia',       'Washington',
   'West Virginia', 'Wisconsin',           'Wyoming' ] ) )
   
# Postal abbreviations of all United States state and territory names
all_us_states_short_trait = Trait( 'TX', [  
   'AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 
   'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 
   'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 
   'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 
   'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VI', 
   'VA', 'WA', 'WV', 'WI', 'WY' ] )
   
#-------------------------------------------------------------------------------
#  Word country related traits:
#-------------------------------------------------------------------------------
   
# Long form of world country names
countries_long_trait = Trait( 'United States', TraitPrefixList( [                   
   'Afghanistan',   'Albania',       'Algeria',     'Andorra',       
   'Angola',        'Antigua and Barbuda',          'Argentina',   
   'Armenia',       'Australia',     'Austria',     'Azerbaijan',  
   'Bahamas',       'Bahrain',       'Bangladesh',  'Barbados',     
   'Belarus',       'Belgium',       'Belize',      'Benin', 
   'Bhutan',        'Bolivia',       'Bosnia and Herzegovina',       
   'Botswana',      'Brazil',        'Brunei',      'Bulgaria',    
   'Burkina Faso',  'Burma/Myanmar', 'Burundi',     'Cambodia',    
   'Cameroon',      'Canada',        'Cape Verde',  'Central African Republic',
   'Chad',          'Chile',         'China',       'Colombia',   
   'Comoros',       'Congo',         'Democratic Republic of Congo', 
   'Costa Rica',    'Cote d\'Ivoire/Ivory Coast',   'Croatia',       
   'Cuba',          'Cyprus',        'Denmark',     'Djibouti',    
   'Dominica',      'Dominican Republic',           'East Timor',  
   'Ecuador',       'Egypt',         'El Salvador', 'Equatorial Guinea', 
   'Eritrea',       'Estonia',       'Ethiopia',    'Fiji',  
   'Finland',       'France',        'Gabon',       'Gambia', 
   'Georgia',       'Germany',       'Ghana',       'Greece', 
   'Grenada',       'Guatemala',     'Guinea',      'Guinea-Bissau', 
   'Guyana',        'Haiti',         'Honduras',    'Hungary', 
   'Iceland',       'India',         'Indonesia',   'Iran',  
   'Iraq',          'Ireland',       'Israel',      'Italy',   
   'Jamaica',       'Japan',         'Jordan',      'Kazakstan', 
   'Kenya',         'Kiribati',      'North Korea', 'South Korea',
   'Kuwait',        'Kyrgyzstan',    'Laos',        'Latvia',
   'Lebanon',       'Lesotho',       'Liberia',     'Libya', 
   'Liechtenstein', 'Lithuania',     'Luxembourg',  'Macedonia', 
   'Madagascar',    'Malawi',        'Malaysia',    'Maldives',  
   'Mali',          'Malta',         'Marshall Islands', 
   'Mauritania',    'Mauritius',     'Mexico',      'Micronesia',    
   'Moldova',       'Monaco',        'Mongolia',    'Morocco',     
   'Mozambique',    'Namibia',       'Nauru',       'Nepal',  
   'Netherlands',   'New Zealand',   'Nicaragua',   'Niger',        
   'Nigeria',       'Norway',        'Oman',        'Pakistan', 
   'Palau',         'Panama',        'Papua New Guinea',   
   'Paraguay',      'Peru',          'Philippines', 'Poland',   
   'Portugal',      'Qatar',         'Romania',    
   'Russian Federation East of the Ural Mountains', 
   'Russian Federation West of the Ural Mountains', 'Rwanda', 
   'Saint Kitts and Nevis',          'Saint Lucia', 
   'Saint Vincent and the Grenadines',              'Samoa',    
   'San Marino',   'Sao Tome and Principe',         'Saudi Arabia', 
   'Senegal',      'Seychelles',     'Sierra Leone',
   'Singapore',    'Slovakia',       'Slovenia',    'Solomon Islands',  
   'Somalia',      'South Africa',   'Spain',       'Sri Lanka',   
   'Sudan',        'Suriname',       'Swaziland',   'Sweden', 
   'Switzerland',  'Syria',          'Taiwan',      'Tajikistan', 
   'Tanzania',     'Thailand',       'Togo',        'Tonga',     
   'Trinidad and Tobago',            'Tunisia',     'Turkey',  
   'Turkmenistan', 'Tuvalu',         'Uganda',      'Ukraine', 
   'United Arab Emirates',           'United Kingdom',  
   'United States',                  'Uruguay',     'Uzbekistan',   
   'Vanuatu',      'Vatican City',   'Venezuela',   'Vietnam',    
   'Yemen',        'Yugoslavia',     'Zambia',      'Zimbabwe' ] ) )
   
#-------------------------------------------------------------------------------
#  Calendar related traits:
#-------------------------------------------------------------------------------

# Long form of month names
month_long_trait = Trait( 'January', TraitPrefixList( [
   'January', 'February', 'March',     'April',   'May',      'June',
   'July',    'August',   'September', 'October', 'November', 'December' ] ),
   cols = 2 )

# Short form of month names
month_short_trait = Trait( 'Jan', [
   'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ], cols = 2 )

# Long form of day of week names
day_of_week_long_trait = Trait( 'Sunday', TraitPrefixList( [
   'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
   'Saturday' ] ), cols = 1 )    

# Short form of day of week names
day_of_week_short_trait = Trait( 'Sun', [
   'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat' ], cols = 1 )    
   
#-------------------------------------------------------------------------------
#  Telephone Number related traits:
#-------------------------------------------------------------------------------

# Local United States phone number
phone_short_trait = Trait( '555-1212',
                           TraitString( regex = r'^\d{3,3}[ -]?\d{4,4}$' ) )
                           
# Long distance United States phone number:                           
phone_long_trait = Trait( '800-555-1212', TraitString( 
                          regex = r'^\d{3,3}[ -]?\d{3,3}[ -]?\d{4,4}$|'
                                  r'^\(\d{3,3}\) ?\d{3,3}[ -]?\d{4,4}$' ) )                         
   
#-------------------------------------------------------------------------------
#  Miscellaneous traits:
#-------------------------------------------------------------------------------

# United States Social Security Number
ssn_trait = Trait( '000-00-0000', 
                   TraitString( regex = r'^\d{3,3}[ -]?\d{2,2}[ -]?\d{4,4}$' ) )

#-------------------------------------------------------------------------------
#  If run from the command line, add all traits to the master traits data base:  
#-------------------------------------------------------------------------------
                                      
if __name__ == '__main__':
    
    from enthought.traits.api import tdb
    define = tdb.define
    
    define( 'zipcode_5',           zipcode_5_trait )
    define( 'zipcode_9',           zipcode_9_trait )
    define( 'us_states_long',      us_states_long_trait )
    define( 'us_states_short',     us_states_short_trait )
    define( 'all_us_states_long',  all_us_states_long_trait )
    define( 'all_us_states_short', all_us_states_short_trait )
    define( 'countries_long',      countries_long_trait )
    define( 'month_long',          month_long_trait )
    define( 'month_short',         month_short_trait )
    define( 'day_of_week_long',    day_of_week_long_trait )
    define( 'day_of_week_short',   day_of_week_short_trait )
    define( 'phone_short',         phone_short_trait )
    define( 'phone_long',          phone_long_trait )
    define( 'ssn_trait',           ssn_trait )
    
