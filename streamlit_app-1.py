from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


st.set_page_config(
    page_title="KIVA Microfinance Repayment Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent


ARTIFACT_DIR = BASE_DIR / "kiva_dashboard_artifacts"


REQUIRED_FILES = {
    "models": ARTIFACT_DIR / "optimized_models.joblib",
    "results": ARTIFACT_DIR / "optimized_results.joblib",
    "grid": ARTIFACT_DIR / "grid_search_summary.joblib",
    "importance": ARTIFACT_DIR / "permutation_importance.joblib",
    "grouped_importance": ARTIFACT_DIR / "grouped_permutation_importance.joblib",
    "consensus": ARTIFACT_DIR / "robust_consensus.joblib",
    "schema": ARTIFACT_DIR / "input_schema.joblib",
    "eda": ARTIFACT_DIR / "eda_bundle.joblib",
}


FEATURE_LABELS = {
    "description.languages": "Loan Description Language",
    "funded_amount": "Funded Amount (USD)",
    "activity": "Loan Activity",
    "sector": "Business Sector",
    "location.country_code": "Country Code",
    "location.country": "Country",
    "location.town": "Town",
    "location.geo.level": "Location Detail Level",
    "partner_id": "Field Partner ID",
    "borrower_gender_group": "Gender",
    "borrowers.pictured": "Borrower Photo",
    "terms.disbursal_amount": "Disbursed Amount (Local Currency)",
    "terms.disbursal_currency": "Disbursal Currency",
    "terms.loan_amount": "Loan Amount (USD)",
    "terms.loss_liability.nonpayment": "Nonpayment Loss Liability",
    "terms.loss_liability.currency_exchange":
        "Currency Exchange Loss Liability",
    "lat": "Latitude",
    "lon": "Longitude",
    "posted_year": "Posting Year",
    "posted_month": "Posting Month",
    "posted_dayofweek": "Posting Day of Week",
    "posted_quarter": "Posting Quarter",
    "disbursal_year": "Disbursal Year",
    "disbursal_month": "Disbursal Month",
    "disbursal_dayofweek": "Disbursal Day of Week",
}

FEATURE_HELP = {
    "borrowers.pictured": "Whether the borrower supplied a profile photo.",
    "partner_id": "Identifier of the Kiva field partner handling the loan.",
    "posted_dayofweek": "0 = Monday and 6 = Sunday.",
    "disbursal_dayofweek": "0 = Monday and 6 = Sunday.",
}

CATEGORY_DISPLAY_NAMES = {
    "description.languages": {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "id": "Indonesian",
        "ru": "Russian",
    },
    "borrower_gender_group": {
        "m": "Male",
        "f": "Female",
    },
    "location.geo.level": {
        "country": "Country-level coordinates",
        "town": "Town-level coordinates",
    },
    "terms.loss_liability.nonpayment": {
        "lender": "Lender",
        "partner": "Field partner",
    },
    "terms.loss_liability.currency_exchange": {
        "lender": "Lender",
        "partner": "Field partner",
        "shared": "Shared",
    },
}

LOCATION_FEATURES = {
    "location.country",
    "location.town",
    "location.country_code",
}
MISSING_OPTION = "<missing>"

LOCATION_LOOKUP = {'afghanistan': {'code': 'af', 'towns': ['jalalabad', 'kabul']},
 'azerbaijan': {'code': 'az',
                'towns': ['absheron region',
                          'agsu',
                          'baku',
                          'baku, city',
                          'beylagan',
                          'bilasuvar',
                          'devechi',
                          'fizuli district',
                          'fuzuli region',
                          'imishli',
                          'khachmaz',
                          'khachmaz region',
                          'khachmaz town',
                          'khirdalan settlement',
                          'khirdalan town',
                          'khudat town',
                          'saatli',
                          'saatli town',
                          'sabirabad',
                          'salyan region',
                          'sumgayit city']},
 'benin': {'code': 'bj', 'towns': ['cotonou']},
 'bolivia': {'code': 'bo',
             'towns': ['chahuira-pampa',
                       'cochabamba',
                       'el alto',
                       'el alto / la paz',
                       'huayrocondo',
                       'la paz',
                       'la paz / el alto',
                       'santa cruz',
                       'santa cruz de la sierra']},
 'bosnia and herzegovina': {'code': 'ba',
                            'towns': ['banovi\x07i',
                                      'doboj',
                                      'gra\nanica',
                                      'kalesija',
                                      'klokotnica, doboj',
                                      'kolesija',
                                      'lukavac',
                                      'sarajevo']},
 'bulgaria': {'code': 'bg',
              'towns': ['asenovgrad',
                        'blagoevgrad',
                        'kardjali',
                        'razgrad',
                        'silistra',
                        'sliven',
                        'smolian',
                        'vratza']},
 'cambodia': {'code': 'kh',
              'towns': ['an doung samrith village',
                        'andong village',
                        'angk village',
                        'angkrang village',
                        'angkrorng village',
                        'banteay meanchey',
                        'batambong',
                        'battambang',
                        'beong veng village',
                        'bro vash village',
                        'choeung wat village',
                        'chom teav village',
                        'chreaek village',
                        'chrey bet meas village',
                        'chrork rom deng village',
                        'chrouy dang village',
                        'da village',
                        'damnak rovieng village',
                        'damnak sangke village',
                        'dangkao district',
                        'dei edth kaoh phos village',
                        'kampong cham',
                        'kampong cham province',
                        'kampong chhnang province',
                        'kampong chnang province',
                        'kampong chrey village',
                        'kandal province',
                        'kandal steung district',
                        'kandal village',
                        'kanthorm village',
                        'kbal damrey village',
                        'kdey tnaut village',
                        'kean svay district',
                        'khsach kandal district',
                        'kompong cham',
                        'kompong tom',
                        'korng keb village',
                        'kouk trob village',
                        'kraing sromor village',
                        'kraing svay village',
                        'kruos village',
                        'kvav village',
                        'leap village',
                        'moen tomrang village',
                        'muk kampoul district',
                        'phnom penh',
                        'phnom penh city',
                        'phum dabprambei village',
                        'phum por village',
                        'phum soupy village',
                        'pon-nhea leu district',
                        'pong toek village',
                        'ponlai village',
                        'pou doh village',
                        'preaek thum village',
                        'prey chhor, kampong cham',
                        'prey khla tboung village',
                        'prey veng province',
                        'preyveng province',
                        'pursat',
                        'roneam tnort village',
                        'rotheng',
                        'sa ang, phnom penh',
                        'sambuor village',
                        'sdao kanlaeng village',
                        'siem reap',
                        'siem reap province',
                        'siemreap province',
                        'slab ta aon village',
                        'slaeng village',
                        'sre khtom village',
                        'sya village',
                        'ta keb village',
                        'ta khmao district',
                        'ta pen village',
                        'ta preak village',
                        'ta skaom village',
                        'taing roleang village',
                        'taing russey village',
                        'takeo province',
                        'tboung khmum district',
                        'thmey village',
                        'thommoney village',
                        'trapaing angkrorng village',
                        'trapaing ronas village',
                        'trapang mates village',
                        'trapeang krasang village']},
 'cameroon': {'code': 'cm',
              'towns': ['bamenda food market, nw. province',
                        'behind pmi',
                        'foncha st, nkwen, nw. province',
                        'food-market, bamenda, nw. province',
                        'guneku, mbengwi, nw province',
                        'mbengwi, nw. province',
                        'ndamukong, bamenda, nw. province.',
                        'nkwen, bamenda, nw province',
                        'ntaghem, bamenda, nw. province',
                        'ntahsen market square, nw. province',
                        'rtc fonta, bambui, nw. province']},
 "cote d'ivoire": {'code': 'ci', 'towns': ['abobo', 'adjamé-forum', 'yopougon']},
 'dominican republic': {'code': 'do',
                        'towns': ['baní',
                                  'consuelo',
                                  'cotuí',
                                  'don juan community of yamasá',
                                  'el seybo',
                                  'hato mayor',
                                  'la romana',
                                  'los alcarrizos',
                                  'mulo community of yamasa',
                                  'palamara community of san cristobal',
                                  'puerto plata',
                                  'samana',
                                  'san cristobal',
                                  'san isidro, community of santo domingo',
                                  'san luis, community of santo domingo',
                                  'san pedro de macorís',
                                  'santiago',
                                  'santo domingo',
                                  'yamasa']},
 'ecuador': {'code': 'ec',
             'towns': ['baba', 'cuenca', 'guayaquil', 'isidro ayora', 'salitre', 'santa lucia']},
 'gaza': {'code': None, 'towns': ['rafah']},
 'ghana': {'code': 'gh',
           'towns': ['abokobi, accra',
                     'abura',
                     'accra',
                     'adenta, accra',
                     'amanfrom, kasoah',
                     'cape coast',
                     'dodowa, dangme west district,',
                     'dorma',
                     'dorma ahenklro',
                     'dormaa',
                     'effiakuma, takoradi',
                     'elmina',
                     'goaso',
                     'kasoa',
                     'katapo,pokuase,ga west district',
                     'koforidua',
                     'mampong',
                     'mankessim',
                     'nkawkaw',
                     'nkurankan, yilo krobo district,',
                     'obuasi',
                     'oda',
                     'offinsu',
                     'pantang',
                     'sefwi wiawso',
                     'sekondi',
                     'sekondi-takoradi',
                     'sesemi, accra',
                     'shama,',
                     'siwdo',
                     'somanya',
                     'suhum',
                     'sunyani',
                     'tarkwa',
                     'tema',
                     'wiawso']},
 'guatemala': {'code': 'gt',
               'towns': ['nahualá, sololá',
                         'paxixil, totonicapan',
                         'poxlajuj, totonicapan',
                         'retalhuleu',
                         'san felipe retalhuleu',
                         'san pedro cutzan, mazatenango',
                         'santo domingo xenacoj, chimaltenango']},
 'haiti': {'code': 'ht', 'towns': ['trou-du-nord']},
 'honduras': {'code': 'hn',
              'towns': ['choluteca',
                        'comayaguela',
                        'danli',
                        'el paraiso',
                        'san lorenzo',
                        'siguatepeque',
                        'tegucigalpa']},
 'indonesia': {'code': 'id',
               'towns': ['badung',
                         'darmaga, bogor,',
                         'legok, tangerang',
                         'leuwiliang, bogor',
                         'leuwisadeng, bogor',
                         'melaya, bali',
                         'pamijahan, bogor',
                         'rajeg, tangerang',
                         'sempidi',
                         'tangerang']},
 'iraq': {'code': 'iq', 'towns': ['kirkuk']},
 'kenya': {'code': 'ke',
           'towns': [': osajai, amagoro,teso district',
                     'amagoro,teso district',
                     'bumula, bungoma',
                     'bungoma',
                     'chemasiri, teso',
                     'isinya',
                     'kangema',
                     'kayole, nairobi',
                     'kiambu',
                     'kibera nairobi',
                     'kibera slums, nairobi',
                     'kisumu',
                     'kitale, mount elgon',
                     'malaba \x13 kamuriai location, teso - kenya',
                     'malaba - teso district',
                     'mayanja,bungoma district',
                     'migori',
                     'mlolongo settlement',
                     'mombasa',
                     'muranga',
                     'myanga, bungoma district',
                     'nairobi',
                     'nairobi city',
                     'nakuru',
                     'nyandarwa',
                     'olturoto, kajiado district',
                     'ongata rongai',
                     'pondamali, nakuru',
                     'rongo',
                     'subukia',
                     'thika town',
                     'ugunja',
                     'wanyororo']},
 'lebanon': {'code': 'lb',
             'towns': ['abbassieh - south',
                       'aley - chouf',
                       'baalback - bekaa',
                       'beirut',
                       'bekaa',
                       'borj barajneh',
                       'nabatieh',
                       'old saida',
                       'saida - south',
                       'sid al bauchrieh - beirut',
                       'tyre, south']},
 'liberia': {'code': 'lr', 'towns': ['lofa co. northern liberia', 'monrovia north']},
 'mali': {'code': 'ml',
          'towns': ['bougouni',
                    'fana koulikoro',
                    'koutiala',
                    'ouelessebougou',
                    'segou',
                    'sikasso',
                    'yanfolila']},
 'mexico': {'code': 'mx',
            'towns': ['acala',
                      'acuna',
                      'altamira, tamaulipas',
                      'apodaca, n. l.',
                      'bochil',
                      'cadereyta, nuevo león, méxico',
                      'chenalho',
                      'escobedo, n. l.',
                      'escobedo, nuevo león',
                      'garcía, nuevo león',
                      'guachochi, chihuahua',
                      'guadalupe, nuevo leon',
                      'hermosillo, sonora',
                      'juárez, n. l.',
                      'larrainzar',
                      'limon # 3324, col. moderna, mty, n.l.',
                      'linares',
                      'monclova,coahuila',
                      'montemorelos, nuevo león',
                      'monterrey',
                      'monterrey, nuevo león',
                      'mérida, yucatán',
                      'nuevo laredo',
                      'ocosingo',
                      'pedro escobedo, querétaro',
                      'piedras negras',
                      'piedras negras, ver.',
                      'reynosa, tamaulipas',
                      'sabinas hidalgo, nuevo león',
                      'saltillo, coahuila',
                      'san cristobal de las casas',
                      'san juanito, chihuahua',
                      'san nicolás de los garza, nuevo león',
                      'santa catarina, nuevo leon',
                      'santo domingo, nuevo leon',
                      'socoltenango',
                      'solidaridad, monterrey',
                      'solistahuacan',
                      'tampico, tamaulipas',
                      'teopisca',
                      'victoria, tamaulipas',
                      'yucatán',
                      'zacatecas',
                      'zinacantan']},
 'moldova': {'code': 'md',
             'towns': ['cahul', 'calarasi', 'floresti', 'gradiste', 'slobozia mare', 'tibirica']},
 'mongolia': {'code': 'mn', 'towns': ['arhangay', 'ulaanbaatar', 'uvurhangay']},
 'mozambique': {'code': 'mz',
                'towns': ['bela-vista, maputo',
                          'boane, maputo',
                          'catembe, maputo',
                          'chissano',
                          'cidade de maputo',
                          'infulene-maputo',
                          'laulane-maputo',
                          'magoanine-maputo',
                          'namaacha',
                          'xai-xai']},
 'nepal': {'code': 'np',
           'towns': ['bhatkepati',
                     'bholdhoka',
                     'dhalko',
                     'koteshwor',
                     'kritpur',
                     'kulashwor',
                     'sainbubhaishepati',
                     'sanepa',
                     'sundhara',
                     'thecho']},
 'nicaragua': {'code': 'ni',
               'towns': ['bluefields',
                         'chinandega',
                         'chontales',
                         'esteli',
                         'leon',
                         'managua',
                         'masaya',
                         'nindiri',
                         'rivas']},
 'nigeria': {'code': 'ng',
             'towns': ['abuja',
                       'agbor,delta state',
                       'asaba, delta state',
                       'benin city',
                       'ibadan, oyo state',
                       'ibadan,oyo state',
                       'jos',
                       'lagos state',
                       'ogbomoso, oyo state',
                       'uromi, edo state']},
 'pakistan': {'code': 'pk',
              'towns': ['arifwala',
                        'borewala',
                        'chichawatni',
                        'depalpur',
                        'kasur',
                        'lahore',
                        'multan',
                        'pakpattan',
                        'raiwind',
                        'vehari']},
 'paraguay': {'code': 'py',
              'towns': ['asunción',
                        'caacupe',
                        'caaguazú',
                        'carapegua',
                        'chaco',
                        'ciudad del este',
                        'encarnación',
                        'ita',
                        'mariano r. alonso',
                        'paraguari',
                        'san ignacio',
                        'san lorenzo',
                        'santaní',
                        'villa elisa',
                        'ybycuí']},
 'peru': {'code': 'pe',
          'towns': ['agallpampa, la libertad',
                    'anta',
                    'ayacucho',
                    'banda de shilcayo - san martín',
                    'bellavista - san martin',
                    'callería - ucayali',
                    'carabayllo-lima',
                    'carhuaz - ancash',
                    'chanchamayo',
                    'chao - viru - la libertad',
                    'comas',
                    'condurri, el collao, puno',
                    'cusco',
                    'cuturapi, yunguyo - puno',
                    'el agustino',
                    'el porvenir - trujillo',
                    'el porvenir-trujillo-la libertad',
                    'florencia de mora - la libertad',
                    'huancayo',
                    'huaraz - ancash',
                    'huaycan',
                    'huayllacocha',
                    'ilave, el collao - puno',
                    'juanjui - san martin',
                    'juli, chucuito - puno',
                    'juliaca',
                    'la esperanza - trujillo',
                    'lamas - san martín',
                    'lima',
                    'los olivos',
                    'mache - la libertad',
                    'matucana',
                    'mazocruz, puno',
                    'moche - trujillo - la libertad',
                    'morales - san martín',
                    'moyobamba - san martin',
                    'nueva requena - ucayali',
                    'ollantaytambo',
                    'otuzco - la libertad',
                    'paijan, la libertad',
                    'pichanaki',
                    'picota - san martín',
                    'pilcuyo - collao, puno',
                    'pucallpa',
                    'pucallpa-ucayali',
                    'puente piedra',
                    'puno',
                    'puno, puno',
                    'quiquijana',
                    'rioja, san martin',
                    'san josé de sisa - san martin',
                    'san juan de lurigancho-lima',
                    'san martin de porres',
                    'san martín',
                    'san martín - san martín',
                    'santa rosa, puno',
                    'saposoa - san martín',
                    'tarapoto - san martin',
                    'tarma',
                    'trujillo - la libertad',
                    'victor larco, trujillo - la libertad',
                    'vitarte',
                    'yarinacocha - ucayali',
                    'yungay - ancash',
                    'yunguyo-puno',
                    'zepita, puno']},
 'philippines': {'code': 'ph',
                 'towns': ['aloran, misamis occidental',
                           'arcon tumauini',
                           'bacolod, ozamiz city, mis. occ.',
                           'bago city, negros occidental',
                           'baliangao misamis occidental',
                           'baliangao, misamis occidental',
                           'banga, south cotabato',
                           'banlag, valencia city',
                           'barangay 3, talakag, bukidnon',
                           'barangay santiago, iligan city',
                           'barangay sinili, santiago city',
                           'bianrzang quirino, isabela',
                           'biasong, lopez jaena, misamis occidental',
                           'bimonton mallig, isabela',
                           'brgy. yumbing camiguin',
                           'brgy.roxas, solano,nueva vizcaya',
                           'buliwao, quezon, nueva vizcaya',
                           'buliwao,quezon, nueva vizcaya',
                           'bunawan, agusan del sur',
                           'calamba misamis occidental',
                           'calamba, misamis occidental',
                           'calape, bohol',
                           'calaran,calamba misamis occidental',
                           'camiguin , province',
                           'canicapan, clarin, misamis occidental',
                           'catadman, ozamiz city mis. occidental',
                           'cauayan city, isabela',
                           'cauayan, isabela',
                           'cebulin, plaridel, misamis occidental',
                           'clarin misamis occidental',
                           'clarin, misamis occidental',
                           'clarin,misamis occidental',
                           'comon, mambajao camiguin',
                           'danao, plaridel, misamis occidental',
                           'dapitan city, zamboanga del norte',
                           'dapitan, city',
                           'dawo, dapitan city',
                           'dicayas, dipolog city',
                           'dipolog city',
                           'dipolog city, zamboanga del norte',
                           'divisoria santiago city',
                           'dumalinao, zamboanga del sur',
                           'fatima, general santos city',
                           'ilagan public market',
                           'ilagan, isabela',
                           'isabela',
                           'kapatungan, agusan del sur',
                           'la paz, cabatuan isabela',
                           'lagag, sindangan, zamboanga del norte',
                           'laligan, valencia city, bukidnon',
                           'libertad bajo,sinacaban,misamis occidental',
                           'liwayway diffun quirino',
                           'loon, bohol',
                           'lopez jaena misamis occidental',
                           'lopez jaena, misamis occidental',
                           'lullutan, ilagan, isabela',
                           'luyong bonbon opol misamis oriental',
                           'macabayao,jimenez,misamis occidental',
                           'magting, mambajao camiguin',
                           'mahayahay, aurora, zamboanga del sur',
                           'malaubang, ozamiz city',
                           'manano, mallig, isabela',
                           'mangidkid, plaridel, misamis occidental',
                           'manil, leon b. postigo, zamboanga del norte',
                           'maribojoc, bohol',
                           'migcanaway, tangub city',
                           'monguia, dupax del norte, nueva vizcaya',
                           'naasag, mambajao camiguin',
                           'naga, jimenez, misamis occidental',
                           'navalan, tukuran, zamboanga del sur',
                           'norala, south cotabato',
                           'nueva vizcaya',
                           'old san mariano, san mariano isabela',
                           'oroquieta city misamis occidental',
                           'oroquieta city, misamis occidental',
                           'owaon, dapitan city',
                           'ozamiz city',
                           'palao solana, cagayan',
                           'panduma, tukuran, zamboanga del sur',
                           'paniki, bagabag, nueva vizcaya',
                           'pines, ororquieta city, misamis occidental',
                           'plaridel, misamis occidental',
                           'poblacion 4, clarin, misamis occidental',
                           'poblacion, diadi, nueva vizcaya',
                           'punit benito soliven, isabela',
                           'puntod lopez jaena,misamis occidental',
                           'purok 9, tag-ibo, dalipuga, iligan city',
                           'purok 9b kiwalan,  iligan city',
                           'quirino, isabela',
                           'rizal santiago city',
                           'rizal, zamboanga del norte',
                           'rizaluna, cordon, isabela',
                           'roxas',
                           'roxas, isabela',
                           'roxas, solano, nueva vizcaya',
                           'san carlos, tukuran, zamboanga del sur',
                           'san isidro bajo, mis. occ.',
                           'san jose, agusan del sur',
                           'san mateo quirino, isabela',
                           'san pablo, alicia, isabela',
                           'santiago , iligan city',
                           'santiago city, isabela',
                           'sapang dalaga, misamis occidental',
                           'sayon, sta. josefa, ads',
                           'senor, sinacaban,misamis occidental',
                           'sibutad',
                           'sibutad, zamboanga del norte',
                           'sikatuna, bohol',
                           'sinacaban,misamis occidental',
                           'sindangan, zamboanga del norte',
                           'solano, nueva vizcaya',
                           'southern pob. plaridel, misamis occidental',
                           'sta. cruz, plaridel, misamis occidental',
                           'sta. josefa, agusan del sur',
                           'sto niño, south cotabato',
                           'sto. nino, south cotabato',
                           'sto.nino, manolo fortich, bukidnon',
                           'sto.nino, south cotabato',
                           'tabo-o, jimenez,misamis occidental',
                           'taboo,jimenez,misamis occidental',
                           'tandul, cabatuan isabela',
                           'tawi-tawi, aloran, misamis occidental',
                           'tinaclaan,clarin,misamis occidental',
                           'tinago, ozamiz city',
                           'trento, agusan del sur',
                           'tubod, aloran, misamis occidental',
                           'tuguegarao city, cagayan',
                           'tukuran, zamboanga del norte',
                           'tumauini, isabela',
                           'tumpagon, cagayan de oro city',
                           'tumpagon,cagayan de oro city',
                           'uddiawan, solano, nueva vizcaya',
                           'ugad cabagan, isabela',
                           'upper bunawan, calamba misamis occidental',
                           'veruela, agusan del sur',
                           'villa coloma, bagabag, nueva vizcaya']},
 'rwanda': {'code': 'rw', 'towns': ['kicukiro/kigali']},
 'samoa': {'code': 'ws',
           'towns': ['apolima',
                     'aufaga',
                     'elise fou',
                     'faga, savaii island',
                     'fagalii uta',
                     'fagasa',
                     'faleasiu',
                     'falelauniu',
                     'faleseela',
                     'faleseela, lefaga',
                     'faleula',
                     'fogatuli savaii',
                     'fusi safata',
                     'gagaifo',
                     'lalovea',
                     'leloto leauvaa',
                     'lepale fasitoo',
                     'letogo',
                     'levi saleimoa',
                     'lotofaga aleipata',
                     'maagiagi',
                     'malie',
                     'maninoa',
                     'manono',
                     'matafaa',
                     'moamoa',
                     'moamoa k tai',
                     'mulifanua',
                     'mulivai leauvaa',
                     'mutiatele',
                     'nofoalii',
                     'nonoa',
                     'papauta',
                     'poutasi',
                     'puipaa',
                     'saanapu tai',
                     'sagafili, mulifanua',
                     'saleaumua, aleipata',
                     'saleilua',
                     'salepouae',
                     'salesatele',
                     'saletele',
                     'samalaeulu leauvaa',
                     'samatau',
                     'samatau fou',
                     'samusu',
                     'satapuala',
                     'satuiatua savaii',
                     'sili, savaii island',
                     'sinamoga',
                     'siumu',
                     'tafangamanu',
                     'taga, savaii island',
                     'taufusi',
                     'tiavea',
                     'toamua',
                     'tuanai',
                     'tulaele',
                     'vaialua nofoalii',
                     'vaigaga',
                     'vailele',
                     'vailoa uta',
                     'vaimoso',
                     'vaitele',
                     'vaitoloa',
                     'vaiusu']},
 'senegal': {'code': 'sn',
             'towns': ['baghagha',
                       'bayouf pout ; thiès',
                       'bignona,ziguinchor',
                       'diourbel',
                       'goudomp',
                       'karsia, kolda',
                       'kolda',
                       'kounkané,kolda',
                       'loudia ouolof',
                       'mboul',
                       'mbour',
                       'ndiaye ndiaye',
                       'ndoucoura thiès',
                       'thiadiaye',
                       'thies',
                       'thiékène; thiès',
                       'ziguinchor']},
 'sierra leone': {'code': 'sl', 'towns': ['kabala', 'magburaka, tonkolili', 'makeni']},
 'south sudan': {'code': None,
                 'towns': ['atlabara,juba,southern',
                           'buluk, juba, southern',
                           'jebel kujur,juba',
                           'munuki, juba, southern']},
 'tajikistan': {'code': 'tj',
                'towns': ['abdurahmon jomi',
                          'asht',
                          'bohtar',
                          'chkalovsk',
                          'dushanbe',
                          'fayzabad',
                          'gafurov',
                          'gafurov, tajikistan',
                          'gissar',
                          'gonjy',
                          'isfara',
                          'istaravshan',
                          'j.rasulov',
                          'j.rumy',
                          'kairakkum',
                          'kanibadam',
                          'khujand',
                          'khuroson',
                          'mastchoh',
                          'nau',
                          'qurgan-tube',
                          'rudaki',
                          'shahrituz',
                          'spitamen',
                          'tursun-zoda',
                          'tursunzade',
                          'utkansoi',
                          'vahdat',
                          'vakhsh',
                          'varzob',
                          'yavan']},
 'tanzania': {'code': 'tz',
              'towns': ['dar es salaam',
                        'kyela',
                        'mbeya',
                        'morogoro',
                        'mwanza',
                        'tanga',
                        'zanzibar']},
 'the democratic republic of the congo': {'code': 'cd', 'towns': ['kinshasa']},
 'togo': {'code': 'tg',
          'towns': ['adidogomé',
                    'adéta',
                    'afagnan',
                    'agbanakin',
                    'agoe',
                    'agome-glozou (aného)',
                    'amégnran',
                    'anfoin',
                    'aného',
                    'assahoun',
                    'avetonou',
                    'awavé (aného)',
                    'celine simisi',
                    'danyi',
                    'gapé',
                    'kpalimé',
                    'lavie',
                    'lomé',
                    'sanguéra',
                    'tabligbo',
                    'tsévié',
                    'wome']},
 'uganda': {'code': 'ug',
            'towns': ['banda-kireka',
                      'bugiri',
                      'bukoto, ntinda',
                      'bushenyi',
                      'ibanda',
                      'iganga',
                      'jinja',
                      'jjanya',
                      'kampala',
                      'kasanje',
                      'kasubi',
                      'kawuku',
                      'kayunga',
                      'kireka',
                      'kitintale',
                      'kiyinde',
                      'kyengara',
                      'kyengera',
                      'lugazi',
                      'makindye',
                      'mityana',
                      'mukono',
                      'naalya',
                      'nansana',
                      'nateete',
                      'ndeeba',
                      'ntungamo',
                      'pallisa',
                      'zana']},
 'ukraine': {'code': 'ua',
             'towns': ['berdyansk',
                       'beregovo',
                       'drogobych',
                       'ivano-frankivsk',
                       'kahovka',
                       'kharkiv',
                       'kherson',
                       'melitopol',
                       'mikhailovka',
                       'mukachevo',
                       'nikolaev',
                       'nikopol',
                       'novaya kahovka',
                       'novomoskovsk',
                       'ordzhonikidze',
                       'pavlograd',
                       'simferopol',
                       'uzhgorod',
                       'vinogradovo',
                       'zaporozhye']},
 'viet nam': {'code': 'vn',
              'towns': ['bac ninh',
                        'do luong',
                        'dong anh- ha noi',
                        'duc linh, binh thuan',
                        'ham thuan nam',
                        'hung nguyen',
                        'kim dong',
                        'long my, hau giang',
                        'me linh',
                        'nghi loc',
                        'ngoc thanh',
                        'ninh giang',
                        'phuc yen',
                        'phung hiep,hau giang',
                        'quang xuong',
                        'soc son',
                        'tanh linh',
                        'y yen',
                        'yen lac']}}


def feature_label(feature):
    """Return a readable label without changing the model's column name."""
    return FEATURE_LABELS.get(
        feature,
        feature.replace(".", " ").replace("_", " ").title(),
    )


def display_category(feature, value):
    """Format a category for people while preserving its raw model value."""
    if value == MISSING_OPTION:
        return "Not specified"

    raw_value = str(value)
    mapped_value = CATEGORY_DISPLAY_NAMES.get(feature, {}).get(raw_value)
    if mapped_value is not None:
        return mapped_value

    if feature in {"location.country_code", "terms.disbursal_currency"}:
        return raw_value.upper()


    printable_value = "".join(
        character if character.isprintable() else " "
        for character in raw_value
    )
    return " ".join(printable_value.replace("_", " ").split()).title()


def numeric_input_kwargs(feature, default):
    """Use integer controls for identifier and calendar fields."""
    integer_features = {
        "partner_id",
        "posted_year",
        "posted_month",
        "posted_dayofweek",
        "posted_quarter",
        "disbursal_year",
        "disbursal_month",
        "disbursal_dayofweek",
    }

    if feature in integer_features:
        return {
            "value": int(round(float(default))),
            "step": 1,
            "format": "%d",
        }

    return {"value": float(default)}


def friendly_feature_table(table):
    """Show readable feature names in dashboard tables and charts."""
    friendly_table = table.copy()
    if "feature" in friendly_table.columns:
        friendly_table["feature"] = friendly_table["feature"].map(feature_label)
    return friendly_table

missing_files = [str(p) for p in REQUIRED_FILES.values() if not p.exists()]

if missing_files:
    st.error(
        "Model artifacts are missing. Run PART 27–30 in the notebook first, "
        "then place this app beside the kiva_dashboard_artifacts folder."
    )
    st.code("\n".join(missing_files))
    st.stop()

models = joblib.load(REQUIRED_FILES["models"])
results_df = joblib.load(REQUIRED_FILES["results"])
grid_df = joblib.load(REQUIRED_FILES["grid"])
permutation_tables = joblib.load(REQUIRED_FILES["importance"])
grouped_tables = joblib.load(REQUIRED_FILES["grouped_importance"])
consensus_df = joblib.load(REQUIRED_FILES["consensus"])
schema = joblib.load(REQUIRED_FILES["schema"])
eda = joblib.load(REQUIRED_FILES["eda"])

st.title("KIVA Microfinance Loan Repayment Prediction")
st.caption(
    "Five optimized machine-learning approaches trained with preprocessing + "
    "SMOTE inside cross-validation. Predictions are decision-support outputs, "
    "not causal conclusions."
)

tab_predict, tab_eda, tab_models, tab_importance = st.tabs(
    ["Predict", "EDA", "Model comparison", "Feature importance"]
)

with tab_predict:
    st.subheader("Loan repayment prediction")

    model_name = st.selectbox(
        "Choose a model",
        options=list(models.keys())
    )

    st.info(
        "Enter information that is known at screening time. "
        "The selected fitted pipeline automatically applies the same preprocessing "
        "used during training."
    )

    user_values = {}

    # Regular widgets are used instead of st.form so changing Country reruns
    # the page immediately and refreshes the Town options.
    with st.container(border=True):
        st.markdown("#### Loan and borrower inputs")

        columns = st.columns(2)
        handled_features = set()
        visible_index = 0

        for feature in schema["feature_columns"]:
            if feature in handled_features:
                continue

            # Country, town and country code are one linked input group. The
            # country code is supplied to the model but is never shown as a
            # separate field that could contradict the selected country.
            if feature in LOCATION_FEATURES:
                country_options = schema["categorical_options"].get(
                    "location.country",
                    [],
                )

                country_container = columns[visible_index % 2]
                town_container = columns[(visible_index + 1) % 2]

                with country_container:
                    selected_country = st.selectbox(
                        feature_label("location.country"),
                        options=country_options,
                        format_func=lambda value: display_category(
                            "location.country",
                            value,
                        ),
                        key="cat_location_country",
                    )

                location_details = LOCATION_LOOKUP.get(
                    selected_country,
                    {"code": None, "towns": []},
                )
                schema_towns = set(
                    schema["categorical_options"].get("location.town", [])
                )
                town_options = sorted(
                    town
                    for town in location_details["towns"]
                    if town in schema_towns
                )
                town_options.append(MISSING_OPTION)

                with town_container:
                    selected_town = st.selectbox(
                        feature_label("location.town"),
                        options=town_options,
                        format_func=lambda value: display_category(
                            "location.town",
                            value,
                        ),
                        key=(
                            "cat_location_town_"
                            f"{country_options.index(selected_country)}"
                        ),
                    )

                country_code = location_details["code"]
                valid_country_codes = set(
                    schema["categorical_options"].get(
                        "location.country_code",
                        [],
                    )
                )

                user_values["location.country"] = selected_country
                user_values["location.town"] = (
                    np.nan
                    if selected_town == MISSING_OPTION
                    else selected_town
                )
                user_values["location.country_code"] = (
                    country_code
                    if country_code in valid_country_codes
                    else np.nan
                )

                handled_features.update(LOCATION_FEATURES)
                visible_index += 2
                continue

            container = columns[visible_index % 2]
            visible_index += 1

            with container:
                if feature == "borrower_gender_group":
                    gender_options = [
                        value
                        for value in schema["categorical_options"].get(
                            feature,
                            [],
                        )
                        if value in {"m", "f"}
                    ]
                    user_values[feature] = st.selectbox(
                        feature_label(feature),
                        options=gender_options,
                        format_func=lambda value: display_category(
                            "borrower_gender_group",
                            value,
                        ),
                        key=f"cat_{feature}",
                    )

                elif feature == "borrowers.pictured":
                    user_values[feature] = st.selectbox(
                        feature_label(feature),
                        options=[1, 0],
                        format_func=lambda value: (
                            "Yes — photo provided"
                            if value == 1
                            else "No — no photo provided"
                        ),
                        help=FEATURE_HELP.get(feature),
                        key=f"flag_{feature}",
                    )

                elif feature in schema["categorical_features"]:
                    options = list(
                        schema["categorical_options"].get(feature, [])
                    )
                    options.append(MISSING_OPTION)

                    selected = st.selectbox(
                        feature_label(feature),
                        options=options,
                        format_func=lambda value, current_feature=feature:
                            display_category(current_feature, value),
                        help=FEATURE_HELP.get(feature),
                        key=f"cat_{feature}",
                    )

                    user_values[feature] = (
                        np.nan if selected == MISSING_OPTION else selected
                    )

                else:
                    default = schema["numeric_defaults"].get(feature, 0.0)

                    # Number inputs stay flexible because a legitimate future
                    # amount may be outside the historical training range.
                    user_values[feature] = st.number_input(
                        feature_label(feature),
                        help=FEATURE_HELP.get(feature),
                        key=f"num_{feature}",
                        **numeric_input_kwargs(feature, default),
                    )

        predict_clicked = st.button(
            "Predict repayment risk",
            type="primary"
        )

    if predict_clicked:
        input_df = pd.DataFrame(
            [user_values],
            columns=schema["feature_columns"]
        )

        estimator = models[model_name]
        prediction = int(estimator.predict(input_df)[0])

        if hasattr(estimator, "predict_proba"):
            default_probability = float(estimator.predict_proba(input_df)[0, 1])
        else:
            score = float(estimator.decision_function(input_df)[0])
            default_probability = 1.0 / (1.0 + np.exp(-score))

        repayment_probability = 1.0 - default_probability

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted outcome",
            "Default risk" if prediction == 1 else "Paid / lower risk"
        )
        c2.metric(
            "Estimated default probability",
            f"{default_probability:.1%}"
        )
        c3.metric(
            "Estimated repayment probability",
            f"{repayment_probability:.1%}"
        )

        st.progress(
            min(max(default_probability, 0.0), 1.0),
            text="Predicted default probability"
        )

        st.caption(
            "This probability is produced by the selected statistical model. "
            "It should not be interpreted as certainty or as a causal judgement "
            "about an individual borrower."
        )


with tab_eda:
    st.subheader("Exploratory data analysis")

    target_counts = pd.Series(eda["target_counts"], name="Loans")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Completed-loan outcomes")
        fig, ax = plt.subplots(figsize=(6, 4))
        target_counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of unique loans")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

    with c2:
        st.markdown("#### Loan amount distribution")
        loan_amount = pd.Series(eda.get("loan_amount", []), dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(loan_amount.dropna(), bins=30)
        ax.set_xlabel("Loan amount")
        ax.set_ylabel("Number of loans")
        st.pyplot(fig)

    sector_data = pd.DataFrame(eda.get("sector_summary", []))

    if not sector_data.empty:
        st.markdown("#### Default rate by sector")

        plot_sector = (
            sector_data
            .sort_values("default_rate_percent", ascending=False)
            .head(15)
            .sort_values("default_rate_percent")
        )
        plot_sector = plot_sector.copy()
        plot_sector["Business Sector"] = plot_sector["sector"].map(
            lambda value: display_category("sector", value)
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            plot_sector["Business Sector"],
            plot_sector["default_rate_percent"]
        )
        ax.set_xlabel("Default rate (%)")
        st.pyplot(fig)

        st.caption(
            "Sector rates are descriptive associations and can be unstable for "
            "small groups. Review loan counts alongside rates."
        )

    country_data = pd.DataFrame(eda.get("country_summary", []))

    if not country_data.empty:
        st.markdown("#### Countries with at least 30 completed loans")

        plot_country = (
            country_data
            .sort_values("default_rate_percent", ascending=False)
            .head(15)
            .sort_values("default_rate_percent")
        )
        plot_country = plot_country.copy()
        plot_country["Country"] = plot_country["location.country"].map(
            lambda value: display_category("location.country", value)
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            plot_country["Country"],
            plot_country["default_rate_percent"]
        )
        ax.set_xlabel("Default rate (%)")
        st.pyplot(fig)

with tab_models:
    st.subheader("Optimized model comparison")

    formatted_results = results_df.copy()

    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        formatted_results[col] = formatted_results[col].map(lambda x: f"{x:.4f}")

    st.dataframe(
        formatted_results,
        width='stretch',
        hide_index=True
    )

    metric_choice = st.selectbox(
        "Metric to compare",
        ["f1", "recall", "precision", "roc_auc", "accuracy"],
        index=0
    )

    plot_df = results_df.sort_values(metric_choice)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["model"], plot_df[metric_choice])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_choice.replace("_", " ").upper())
    ax.set_title(f"Optimized model comparison — {metric_choice.upper()}")
    st.pyplot(fig)

    st.markdown("#### GridSearchCV selected parameters")

    for _, row in grid_df.iterrows():
        with st.expander(row["model"]):
            st.write("Best cross-validation F1:", round(float(row["best_cv_f1"]), 4))
            st.json(row["best_parameters"])


with tab_importance:
    st.subheader("Feature importance")

    st.warning(
        "Feature importance shows predictive association, not causation. "
        "Country, country code, currency, town, latitude and longitude overlap "
        "substantially, so their individual rankings should not be interpreted "
        "as independent effects."
    )

    importance_model = st.selectbox(
        "Choose a model for permutation importance",
        options=list(permutation_tables.keys())
    )

    table = permutation_tables[importance_model].copy()

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=min(25, len(table)),
        value=min(15, len(table))
    )

    top = friendly_feature_table(
        table.head(top_n).sort_values("importance_mean")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"]
    )
    ax.set_xlabel("Mean decrease in test F1 after permutation")
    ax.set_title(f"Permutation importance — {importance_model}")
    st.pyplot(fig)

    st.dataframe(
        friendly_feature_table(table.head(top_n)),
        width='stretch',
        hide_index=True
    )

    grouped = grouped_tables[importance_model].copy()

    st.markdown("#### Conceptual grouped importance")

    plot_grouped = grouped.sort_values("positive_importance")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        plot_grouped["conceptual_group"],
        plot_grouped["positive_importance"]
    )
    ax.set_xlabel("Summed positive permutation importance")
    st.pyplot(fig)

    st.markdown("#### Cross-model robust consensus")
    st.dataframe(
        friendly_feature_table(consensus_df.head(20)),
        width='stretch',
        hide_index=True
    )

    st.caption(
        "Permutation importance can still be diluted when predictors are correlated. "
        "Use the grouped view and model-specific results together in the dissertation."
    )
