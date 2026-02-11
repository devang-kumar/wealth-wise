import requests
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse, Http404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.db.models import Avg
from .models import FinancialPrediction
from .ml_model import predict_finances
from .mlmodel1 import predict_investment
from .models import FinancialPrediction
from django.db.models import Avg, Sum

def recommend_investment(request):
    context = {}
    if request.method == 'POST':
        user_input = {
            'Gender': request.POST['gender'],
            'Age': int(request.POST['age']),
            'Working_professional': request.POST['working_professional'],
            'Annual_income': int(request.POST['annual_income']),
            'Investment_per_month': int(request.POST['investment_per_month']),
            'Goal_for_investment': request.POST['goal'],
            'Duration_to_save_in_Years_': int(request.POST['duration']),
        }

        recommended, probabilities = predict_investment(user_input)
        context['recommended'] = recommended
        context['probabilities'] = probabilities
        context['submitted'] = True

    return render(request, 'mlapp/recommend_investment.html', context)
# Use API Key from settings or fallback to demo for development
API_KEY = getattr(settings, 'FMP_API_KEY', 'demo')

# Common Endpoints
TREND_ENDPOINTS = {
    'actives': 'https://financialmodelingprep.com/api/v3/stock/actives',
    'gainers': 'https://financialmodelingprep.com/api/v3/stock/gainers',
    'losers':  'https://financialmodelingprep.com/api/v3/stock/losers',
}
KEY_MAP = {
    'actives': 'mostActiveStock',
    'gainers': 'mostGainerStock',
    'losers':  'mostLoserStock',
}

def home_view(request):
    context = {}

    if request.user.is_authenticated:
        predictions = FinancialPrediction.objects.filter(user=request.user).order_by('-created_at')[:5]
        total_predictions = predictions.count()
        avg_savings = predictions.aggregate(Avg('savings_percentage'))['savings_percentage__avg']
        total_savings = predictions.aggregate(Sum('predicted_savings'))['predicted_savings__sum']

        context.update({
            'predictions': predictions,
            'total_predictions': total_predictions,
            'avg_savings': avg_savings,
            'total_savings': total_savings,
        })

    return render(request, 'mlapp/home.html', context)

def explore(request):
    return redirect('stocks')

def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        try:
            User.objects.create_user(username=username, password=password)
            return redirect('login')
        except:
            return render(request, 'mlapp/signup.html', {'error': 'Username already exists'})
    return render(request, 'mlapp/signup.html')

def login_view(request):
    if request.method == 'POST':
        user = authenticate(
            request,
            username=request.POST['username'],
            password=request.POST['password']
        )
        if user:
            login(request, user)
            return redirect('home')
        return render(request, 'mlapp/login.html', {'error': 'Invalid credentials'})
    return render(request, 'mlapp/login.html')

def logout_view(request):
    logout(request)
    return redirect('home')

# @login_required(login_url='login')
def dashboard(request):
    preds = FinancialPrediction.objects.filter(user=request.user)
    latest = preds.order_by('-created_at')[:3]
    total = preds.count()
    avg_sav = preds.aggregate(Avg('savings_percentage'))['savings_percentage__avg'] or 0
    return render(request, 'mlapp/dashboard.html', {
        'predictions': latest,
        'total_predictions': total,
        'avg_savings': avg_sav
    })

# @login_required(login_url='login')
def predict_view(request):
    return render(request, 'mlapp/predict.html')

# @login_required(login_url='login')
def history_view(request):
    preds = FinancialPrediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'mlapp/history.html', {'predictions': preds})
# @login_required(login_url='login')
def predict_view(request):
    return render(request, 'mlapp/predict.html')


# @login_required(login_url='login')
def insights_view(request):
    monthly = FinancialPrediction.objects.filter(user=request.user).extra(
        select={'month': "strftime('%%Y-%%m', created_at)"}
    ).values('month').annotate(
        avg_income=Avg('monthly_income'),
        avg_savings=Avg('predicted_savings')
    ).order_by('month')

    avg_sav = FinancialPrediction.objects.filter(user=request.user).aggregate(
        Avg('savings_percentage'))['savings_percentage__avg'] or 0

    best = FinancialPrediction.objects.filter(user=request.user).extra(
        select={'month': "strftime('%%Y-%%m', created_at)"}
    ).values('month').annotate(
        avg_savings=Avg('savings_percentage')
    ).order_by('-avg_savings').first() or {'month': 'N/A', 'avg_savings': 0}

    return render(request, 'mlapp/insights.html', {
        'monthly_data': monthly,
        'avg_savings': avg_sav,
        'best_month': best,
    })
from django.http import JsonResponse
import joblib
import numpy as np
@csrf_exempt
# @login_required(login_url='login')
def predict_api(request):
    if request.method == "POST":
        try:
            # Validate all required fields
            required_fields = [
                'monthly_income', 
                'family_members',
                'emi_rent',
                'annual_income',
                'qualification',
                'earning_members'
            ]
            
            data = {}
            for field in required_fields:
                value = request.POST.get(field)
                if not value:
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Missing required field: {field}'
                    }, status=400)
                
                # Convert numeric fields
                if field in ['monthly_income', 'emi_rent', 'annual_income']:
                    try:
                        data[field] = float(value)
                    except ValueError:
                        return JsonResponse({
                            'status': 'error',
                            'message': f'Invalid number for {field}'
                        }, status=400)
                elif field in ['family_members', 'earning_members']:
                    try:
                        data[field] = int(value)
                    except ValueError:
                        return JsonResponse({
                            'status': 'error',
                            'message': f'Invalid integer for {field}'
                        }, status=400)
                else:
                    data[field] = value

            # Get prediction from ML model
            result = predict_finances(**data)
            
            # Validate ML model response
            if not result or 'status' not in result:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid response from prediction model'
                }, status=500)

            if result.get('status') == 'error':
                return JsonResponse(result, status=400)

            # Save to database
            FinancialPrediction.objects.create(
                user=request.user,
                monthly_income=data['monthly_income'],
                family_members=data['family_members'],
                emi_rent=data['emi_rent'],
                annual_income=data['annual_income'],
                qualification=data['qualification'],
                earning_members=data['earning_members'],
                predicted_spending=result['predicted_spending'],
                predicted_savings=result['predicted_savings'],
                savings_percentage=result['savings_percentage']
            )

            return JsonResponse({
                'status': 'success',
                'predicted_spending': result['predicted_spending'],
                'predicted_savings': result['predicted_savings'],
                'savings_percentage': result['savings_percentage']
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Server error: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Only POST requests are allowed'
    }, status=405)

from django.core.paginator import Paginator

def explore_stocks(request):
    chosen = request.GET.get('filter', 'actives')
    if chosen not in TREND_ENDPOINTS:
        chosen = 'actives'

    try:
        resp = requests.get(f"{TREND_ENDPOINTS[chosen]}?apikey={API_KEY}", timeout=5)
        raw = resp.json() if resp.status_code == 200 else {}
    except requests.RequestException:
        raw = {}

    data_list = []
    if isinstance(raw, dict) and KEY_MAP[chosen] in raw:
        data_list = raw[KEY_MAP[chosen]]
    elif isinstance(raw, list):
        data_list = raw

    stocks = []
    for item in data_list:
        if len(stocks) >= 100:  # Allow more so pagination can show 30/page
            break
        try:
            price = float(item.get('price', 0))
        except:
            continue
        cr = item.get('changesPercentage', '')
        try:
            change = f"{float(cr):+.2f}"
        except:
            change = cr or ''
        stocks.append({
            'symbol': item.get('ticker') or item.get('symbol', ''),
            'name': item.get('companyName') or item.get('name', ''),
            'price': round(price, 2),
            'exchange': item.get('exchange') or item.get('exchangeShortName', ''),
            'change': change,
            'market_cap': item.get('marketCap'),
            'volume': item.get('volume'),
            'week_high': item.get('yearHigh'),
            'week_low': item.get('yearLow'),
        })

    paginator = Paginator(stocks, 30)  # 30 stocks per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    info = ''
    if not stocks:
        info = (
            f"API key = {API_KEY!r}, "
            f"status={resp.status_code if 'resp' in locals() else 'no-response'}, "
            f"raw_keys={list(raw.keys()) if isinstance(raw, dict) else 'not-dict'}, "
            f"items={len(data_list)}"
        )

    return render(request, 'mlapp/explore.html', {
        'stocks': page_obj,
        'current_filter': chosen,
        'debug_info': info,
        'page_obj': page_obj
    })


def stock_detail(request, symbol):
    quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={API_KEY}"
    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}"
    hist_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=365&apikey={API_KEY}"

    try:
        qdata = requests.get(quote_url).json()
        pdata = requests.get(profile_url).json()
        hdata = requests.get(hist_url).json().get('historical', [])
    except:
        raise Http404(f"Failed to fetch data for {symbol}")

    if not qdata or not pdata:
        raise Http404(f"No data found for '{symbol}'")

    quote = qdata[0]
    profile = pdata[0]
    hist = list(reversed(hdata))  # oldest → newest
    dates = [h['date'] for h in hist]
    closes = [h['close'] for h in hist]

    return render(request, 'mlapp/stock_detail.html', {
        'quote': quote,
        'profile': profile,
        'chart_dates': dates,
        'chart_closes': closes,
    })
CRYPTO_API = "https://financialmodelingprep.com/api/v3/cryptocurrencies"


import requests
from django.shortcuts import render

def explore_crypto(request):
    try:
        url = 'https://min-api.cryptocompare.com/data/top/mktcapfull?limit=100&tsym=USD'
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json().get('Data', [])
        coins = []

        for coin in data:
            info = coin.get('CoinInfo', {})
            raw = coin.get('RAW', {}).get('USD', {})

            image_url = info.get('ImageUrl')
            coins.append({
                'name': info.get('FullName'),
                'symbol': info.get('Name'),
                'image': f"https://www.cryptocompare.com{image_url}" if image_url else '',
                'price': raw.get('PRICE', 'N/A'),
                'market_cap': raw.get('MKTCAP', 'N/A'),
            })

    except Exception as e:
        print(f"Error: {e}")
        coins = []

    return render(request, 'mlapp/explore_crypto.html', {'coins': coins})


import requests
import json
from django.shortcuts import render

# Cache coin symbol to ID mapping
def get_symbol_to_id_map():
    try:
        with open('symbol_id_map.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        url = 'https://api.coingecko.com/api/v3/coins/list'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            mapping = {coin['symbol'].lower(): coin['id'] for coin in data}
            with open('symbol_id_map.json', 'w') as f:
                json.dump(mapping, f)
            return mapping
        return {}

def crypto_detail(request, symbol):
    url = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms={symbol}&tsyms=USD"
    chart_url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit=6"

    response = requests.get(url)
    chart_response = requests.get(chart_url)

    data = response.json().get('RAW', {}).get(symbol, {}).get('USD', {})
    chart_data = chart_response.json().get("Data", {}).get("Data", [])

    labels = [point["time"] for point in chart_data]
    prices = [point["close"] for point in chart_data]

    # Convert Unix timestamps to readable dates
    from datetime import datetime
    labels = [datetime.fromtimestamp(ts).strftime("%b %d") for ts in labels]

    context = {
        "symbol": symbol,
        "data": data,
        "labels": labels,
        "prices": prices
    }
    return render(request, "mlapp/crypto_detail.html", context)




def financial_news(request):
    api_key = 'd867bbe4b3bf47d1942dbc3b698fffd8'
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'stock market OR finance OR investment',
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': api_key,
        'pageSize': 10,
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        articles = data.get('articles', [])
    except Exception as e:
        print("Error fetching news:", e)
        articles = []

    return render(request, 'mlapp/news.html', {'articles': articles})

def sip_calculator(request):
    return render(request, 'mlapp/sip_calculator.html')

import requests
from django.shortcuts import render

def mutual_funds_list(request):
    try:
        response = requests.get("https://api.mfapi.in/mf")
        response.raise_for_status()
        all_funds = response.json()

        # Only show top 20 for now
        top_funds = all_funds[:20]

        fund_data = []
        for fund in top_funds:
            code = fund.get('schemeCode')
            name = fund.get('schemeName')

            nav_response = requests.get(f"https://api.mfapi.in/mf/{code}")
            nav_response.raise_for_status()
            nav_json = nav_response.json()
            nav_data = nav_json.get('data', [{}])

            if nav_data:
                latest = nav_data[0]
                latest_nav = latest.get('nav', 'N/A')
                date = latest.get('date', 'N/A')

                # Simulate additional metadata (normally you'd use a better data source)
                simulated_data = {
                    "fund_type": "Equity" if "Equity" in name else "Debt" if "Debt" in name else "Hybrid",
                    "category": "Large Cap" if "Bluechip" in name else "Flexi Cap",
                    "amc": name.split()[0],  # Crude way to get AMC name
                    "return_1y": f"{round(float(latest_nav) * 0.08, 2)}%",  # Simulate 8% 1Y return
                }

                fund_data.append({
                    "code": code,
                    "name": name,
                    "nav": latest_nav,
                    "date": date,
                    **simulated_data
                })

        return render(request, 'mlapp/mutualfunds.html', {'funds': fund_data})

    except requests.exceptions.RequestException as e:
        print(f"Error fetching mutual fund data: {e}")
        return render(request, 'mlapp/mutualfunds.html', {'funds': []})
